"""Sub-block op-level profile of GRN's visual_forward.

The phase-level profile (`refine_phases.py`) showed 99.5 % of per-step time
is the 28-block visual forward. This script drills inside one block and
measures each operation: pre-attn rms_norm, the three QKV projections,
Q/K norms + RoPE, scaled-dot-product attention, o_proj, post-attn rms_norm,
MLP gate/up/silu/down. Runs the visual forward UN-COMPILED (so each op
fires as its own dispatch and we can time it). Absolute time is ~10-15 %
higher than the compiled production path; the RELATIVE breakdown is the
signal — that's how we find the next optimization target.

We patch a single block's methods with mx.eval boundaries between every
op, run a few visual_forward passes, and aggregate the medians.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from xgrn_mlx.grn import GRN2BMLX, rms_norm, swiglu, silu, fused_rope
from xgrn_mlx.rope import apply_rope, text_rope, visual_rope
from xgrn_mlx.text import load_or_create_text_embeddings_with_key
from xgrn_mlx.schedule import scale_schedule
from xgrn_mlx.run import _model


SUB_OPS = [
    "pre_attn_norm",   # rms_norm(x, input_layernorm)
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_transpose",    # transpose to [B, H, L, D]
    "q_norm",
    "k_norm",
    "apply_rope_q",
    "apply_rope_k",
    "sdpa",            # mx.fast.scaled_dot_product_attention
    "attn_reshape",    # transpose+reshape back
    "o_proj",
    "attn_residual",
    "post_attn_norm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_silu_mul",    # silu(gate) * up
    "mlp_down_proj",
    "mlp_residual",
]


def profiled_block(model: GRN2BMLX, x, block, rope, prefix_k, prefix_v, mask, timings):
    """Instrumented copy of GRN2BMLX.block() with per-op timing."""
    def tick(phase, tensor):
        mx.eval(tensor)
        timings[phase].append(time.perf_counter() - tick.t0)
        tick.t0 = time.perf_counter()
    tick.t0 = time.perf_counter()

    cfg = model.config
    # Pre-attn norm
    h = rms_norm(x, model.w[model.key(block, "input_layernorm.weight")])
    tick("pre_attn_norm", h)

    # QKV projections
    b, l, _ = h.shape
    q = model.block_linear(h, block, "attn.q_proj")
    tick("q_proj", q)
    q = q.reshape(b, l, cfg.num_heads, cfg.head_dim)
    k = model.block_linear(h, block, "attn.k_proj")
    tick("k_proj", k)
    k = k.reshape(b, l, cfg.num_key_value_heads, cfg.head_dim)
    v = model.block_linear(h, block, "attn.v_proj")
    tick("v_proj", v)
    v = v.reshape(b, l, cfg.num_key_value_heads, cfg.head_dim)

    # Q/K/V transpose
    q = mx.transpose(q, (0, 2, 1, 3))
    k = mx.transpose(k, (0, 2, 1, 3))
    v = mx.transpose(v, (0, 2, 1, 3))
    tick("qk_transpose", v)

    # Q-norm, K-norm
    q = rms_norm(q, model.w[model.key(block, "attn.q_norm.weight")])
    tick("q_norm", q)
    k = rms_norm(k, model.w[model.key(block, "attn.k_norm.weight")])
    tick("k_norm", k)

    # RoPE
    q = apply_rope(q, rope)
    tick("apply_rope_q", q)
    k = apply_rope(k, rope)
    tick("apply_rope_k", k)

    # Attention
    cur_k, cur_v = k, v
    if prefix_k is not None:
        k = mx.concatenate([prefix_k, k], axis=2)
        v = mx.concatenate([prefix_v, v], axis=2)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.head_dim ** -0.5, mask=mask)
    tick("sdpa", y)
    y = mx.transpose(y, (0, 2, 1, 3)).reshape(b, l, cfg.embed_dim)
    tick("attn_reshape", y)

    # Output projection + residual
    attn_out = model.block_linear(y, block, "attn.o_proj")
    tick("o_proj", attn_out)
    x = x + attn_out
    tick("attn_residual", x)

    # Post-attn norm
    h = rms_norm(x, model.w[model.key(block, "post_attention_layernorm.weight")])
    tick("post_attn_norm", h)

    # MLP
    gate = model.block_linear(h, block, "mlp.gate_proj")
    tick("mlp_gate_proj", gate)
    up = model.block_linear(h, block, "mlp.up_proj")
    tick("mlp_up_proj", up)
    hidden = silu(gate) * up
    tick("mlp_silu_mul", hidden)
    down = model.block_linear(hidden, block, "mlp.down_proj")
    tick("mlp_down_proj", down)
    x = x + down
    tick("mlp_residual", x)
    return x, cur_k, cur_v


def main():
    print("Loading model + cached text embeddings for t2i-correct shape...", flush=True)
    cond, uncond, key = load_or_create_text_embeddings_with_key(
        "A realistic photo of an orange tabby cat sitting on a windowsill, fluffy fur, green eyes",
        "",
        model_dir=Path("models/GRN"),
        text_dtype="bf16",
        cache_dtype="fp32",
    )
    schedule, mapped = scale_schedule("0.25M", 1.0, 1)
    pt, ph, pw = schedule[0]
    model: GRN2BMLX = _model(
        "t2i", Path("models/GRN"),
        weights_dtype="fp32", compute_dtype="bf16",
        compile_visual_pass=False,   # explicit: no compile so each op fires its own dispatch
    )
    cond_cache = model.encode_text_cache_cached(cond, f"{key}:cond")
    uncond_cache = model.encode_text_cache_cached(uncond, f"{key}:uncond")
    d = model.config.vae_latent_dim * model.config.hbq_round
    pure_rand = mx.random.randint(0, model.config.detail_num_lvl, (1, d, pt, ph, pw), dtype=mx.int32)
    rope = mx.concatenate([visual_rope(pt, ph, pw, mapped), text_rope(1, offset=512)], axis=1)
    mx.eval(rope)
    cfg_cache = model.make_cfg_cache(cond_cache, uncond_cache, pt * ph * pw + 1)
    mixed = pure_rand
    pt_embed = model.pt_embed(0.0)
    embed = model.embed_visual_labels(mixed)
    x = mx.concatenate([embed, pt_embed], axis=1)
    x = mx.concatenate([x, x], axis=0)   # CFG: stack cond + uncond
    mx.eval(x)

    n_warmup_blocks = 28        # 1 full visual forward
    n_measure_steps = 3          # 3 full visual forwards' worth (3 * 28 = 84 block timings each op)

    timings: dict[str, list[float]] = defaultdict(list)
    # Warmup pass — uncompiled, all 28 blocks, but throw timings away.
    print("Warmup pass (1 uncompiled visual forward)...", flush=True)
    warmup_timings: dict[str, list[float]] = defaultdict(list)
    cur_x = x
    for blk in range(28):
        cur_x, _, _ = profiled_block(model, cur_x, blk, rope,
                                     cfg_cache.k[blk], cfg_cache.v[blk], cfg_cache.mask,
                                     warmup_timings)
    print("Measure passes...", flush=True)
    for step in range(n_measure_steps):
        cur_x = x
        for blk in range(28):
            cur_x, _, _ = profiled_block(model, cur_x, blk, rope,
                                         cfg_cache.k[blk], cfg_cache.v[blk], cfg_cache.mask,
                                         timings)

    summary = {}
    for op in SUB_OPS:
        samples = timings[op]
        if not samples:
            continue
        s_sorted = sorted(samples)
        summary[op] = {
            "count": len(samples),
            "median_ms": np.median(samples) * 1000,
            "mean_ms": np.mean(samples) * 1000,
            "total_per_step_ms": np.median(samples) * 28 * 1000,
        }
    # Sum of medians × 28 blocks = approx total uncompiled-visual-forward time per step
    total_per_step_ms = sum(v["total_per_step_ms"] for v in summary.values())
    summary["_meta"] = {
        "warmup_blocks": n_warmup_blocks,
        "measure_steps": n_measure_steps,
        "n_blocks": 28,
        "total_uncompiled_step_ms": total_per_step_ms,
        "production_compiled_step_ms": 1509.66,
    }
    out_path = Path("outputs/track-a/profile/block_ops.json")
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Per-op timing in one GRN block (uncompiled, median over {n_measure_steps * 28} samples) ===")
    print(f"{'op':<22} {'median ms':>10} {'×28 ms':>10} {'pct of step':>13}")
    rows = sorted(
        ((op, v) for op, v in summary.items() if op != "_meta"),
        key=lambda x: -x[1]["total_per_step_ms"],
    )
    for op, v in rows:
        pct = v["total_per_step_ms"] / total_per_step_ms * 100
        print(f"{op:<22} {v['median_ms']:>10.3f} {v['total_per_step_ms']:>10.2f} {pct:>12.2f}%")
    print(f"{'':<22} {'sum (uncompiled)':>10} {total_per_step_ms:>10.2f}")
    print(f"{'':<22} {'production (compiled)':>10} {summary['_meta']['production_compiled_step_ms']:>10.2f}")
    print(f"{'':<22} {'compile speedup':>10} {summary['_meta']['production_compiled_step_ms'] / total_per_step_ms:>10.2f}×")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
