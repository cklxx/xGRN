"""Phase-level instrumentation of GRN refine().

Wraps the inner pieces of `GRN2BMLX.refine`'s per-step loop with mx.eval()
boundaries and a perf_counter timer. Reports median per-phase time over the
50 measured steps (after warmup).

This forces synchronization between phases, which slightly inflates absolute
wall time vs the compiled production path, but the RELATIVE distribution is
the signal we want — it tells us where the biggest non-qkv opt targets are.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from xgrn_mlx.grn import GRN2BMLX
from xgrn_mlx.rope import text_rope, visual_rope
from xgrn_mlx.schedule import refinement_target_pt, scale_schedule
from xgrn_mlx.text import load_or_create_text_embeddings_with_key
from xgrn_mlx.run import _model


PHASES = [
    "visual_input_prep",   # pt_embed + embed_visual_labels + concat
    "visual_forward",      # 28-block CFG visual pass
    "logits_head",         # head rms_norm + head.proj + cfg mix + temperature
    "sampling",            # categorical + reshape + transpose + cast
    "mask_update",         # uniform < target + where + cast
    "mixed_eval",          # mx.eval(mixed) — forces the step to finish
]


def run_one_profile(
    *,
    pn: str = "0.25M",
    steps: int = 50,
    warmup_steps: int = 2,
    h_div_w: float = 1.0,
    prompt: str = (
        "A realistic photo of an orange tabby cat sitting on a windowsill, "
        "fluffy fur, green eyes, soft daylight, natural indoor background, "
        "sharp focus, warm but realistic colors"
    ),
    seed: int = 42,
    guidance: float = 3.0,
    temperature: float = 1.1,
    model_dir: Path = Path("models/GRN"),
) -> dict:
    """Run one t2i-correct-shape refinement loop with phase timing."""
    cond, uncond, prompt_cache_key = load_or_create_text_embeddings_with_key(
        prompt,
        negative_prompt="",
        model_dir=model_dir,
        text_dtype="bf16",
        cache_dtype="fp32",
    )
    num_frames = 1
    schedule, mapped = scale_schedule(pn, h_div_w, num_frames)
    pt, ph, pw = schedule[0]
    model: GRN2BMLX = _model(
        "t2i", model_dir,
        weights_dtype="fp32", compute_dtype="bf16",
        compile_visual_pass=True,
    )
    # Match the path the production refine() takes -- pre-cache text KV.
    cond_cache = model.encode_text_cache_cached(cond, f"{prompt_cache_key}:cond")
    uncond_cache = model.encode_text_cache_cached(uncond, f"{prompt_cache_key}:uncond")
    d = model.config.vae_latent_dim * model.config.hbq_round
    mx.random.seed(seed)
    pure_rand = mx.random.randint(0, model.config.detail_num_lvl, (1, d, pt, ph, pw), dtype=mx.int32)
    mixed = pure_rand
    rope = mx.concatenate([visual_rope(pt, ph, pw, mapped), text_rope(1, offset=512)], axis=1)
    mx.eval(rope)
    cfg_cache = model.make_cfg_cache(cond_cache, uncond_cache, pt * ph * pw + 1)
    target_pts = [refinement_target_pt(s, steps, 1.0) for s in range(steps)]
    token_count = pt * ph * pw
    next_pt = 0.0

    timings: dict[str, list[float]] = defaultdict(list)

    def tick(phase: str, tensor):
        mx.eval(tensor)
        timings[phase].append(time.perf_counter() - tick.t0)
        tick.t0 = time.perf_counter()

    tick.t0 = time.perf_counter()  # type: ignore[attr-defined]

    total_t0 = time.perf_counter()

    pred_labels = pure_rand
    for step in range(warmup_steps + steps):
        cur_pt = next_pt
        # Phase 1: visual input prep
        tick.t0 = time.perf_counter()  # type: ignore[attr-defined]
        pt_embed_token = model.pt_embed(cur_pt)
        embed = model.embed_visual_labels(mixed)
        visual_input = mx.concatenate([embed, pt_embed_token], axis=1)
        # Phase 2: visual forward
        if step >= warmup_steps:
            tick("visual_input_prep", visual_input)
        else:
            mx.eval(visual_input)
            tick.t0 = time.perf_counter()  # type: ignore[attr-defined]
        cfg_out = model.visual_forward_cfg_batched(visual_input, rope, cfg_cache)
        # Phase 3: logits head
        if step >= warmup_steps:
            tick("visual_forward", cfg_out)
        else:
            mx.eval(cfg_out)
            tick.t0 = time.perf_counter()  # type: ignore[attr-defined]
        cfg_logits = model.logits(cfg_out, token_count)
        cond_logits = cfg_logits[:1]
        uncond_logits = cfg_logits[1:2]
        logits = (uncond_logits + guidance * (cond_logits - uncond_logits)) / float(temperature)
        # Phase 4: sampling
        if step >= warmup_steps:
            tick("logits_head", logits)
        else:
            mx.eval(logits)
            tick.t0 = time.perf_counter()  # type: ignore[attr-defined]
        sample = mx.random.categorical(logits.reshape(-1, model.config.detail_num_lvl)).reshape(1, token_count, d)
        labels = sample.reshape(1, pt, ph, pw, d)
        pred_labels = mx.transpose(labels, (0, 4, 1, 2, 3)).astype(mx.int32)
        # Phase 5: mask update
        target_pt = target_pts[step % steps]
        if step >= warmup_steps:
            tick("sampling", pred_labels)
        else:
            mx.eval(pred_labels)
            tick.t0 = time.perf_counter()  # type: ignore[attr-defined]
        mask = mx.random.uniform(shape=pred_labels.shape) < target_pt
        mixed_new = mx.where(mask, pred_labels, pure_rand).astype(mx.int32)
        # Phase 6: mixed eval (closes the per-step loop)
        if step >= warmup_steps:
            tick("mask_update", mixed_new)
        else:
            mx.eval(mixed_new)
            tick.t0 = time.perf_counter()  # type: ignore[attr-defined]
        mixed = mixed_new
        next_pt = target_pt
        if step >= warmup_steps:
            tick("mixed_eval", mixed)
        else:
            mx.eval(mixed)
            tick.t0 = time.perf_counter()  # type: ignore[attr-defined]

    total = time.perf_counter() - total_t0

    summary = {}
    for phase, samples in timings.items():
        samples_sorted = sorted(samples)
        summary[phase] = {
            "count": len(samples),
            "median_ms": np.median(samples) * 1000,
            "mean_ms": np.mean(samples) * 1000,
            "p10_ms": samples_sorted[int(len(samples) * 0.1)] * 1000,
            "p90_ms": samples_sorted[int(len(samples) * 0.9)] * 1000,
            "total_s": float(np.sum(samples)),
        }
    summary["_meta"] = {
        "pn": pn,
        "steps_measured": steps,
        "warmup_steps": warmup_steps,
        "total_loop_s": total,
    }
    return summary


def main() -> None:
    print("Running t2i-correct shape phase profile (50 measured steps, 2 warmup)...", flush=True)
    summary = run_one_profile()
    out_path = Path("outputs/track-a/profile/phases.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    # Pretty print
    print(f"\n=== Per-phase timing (median over {summary['_meta']['steps_measured']} steps) ===")
    print(f"{'phase':<24} {'median (ms)':>12} {'mean (ms)':>10} {'total (s)':>10} {'pct of total':>13}")
    total_of_phases_s = sum(
        v["total_s"] for k, v in summary.items() if k != "_meta"
    )
    for phase in PHASES:
        v = summary[phase]
        pct = v["total_s"] / total_of_phases_s * 100
        print(f"{phase:<24} {v['median_ms']:>12.3f} {v['mean_ms']:>10.3f} {v['total_s']:>10.3f} {pct:>12.1f} %")
    print(f"{'':<24} {'sum of phases:':<33} {total_of_phases_s:>10.3f}")
    print(f"{'':<24} {'loop wall:':<33} {summary['_meta']['total_loop_s']:>10.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
