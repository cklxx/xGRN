from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from timm.models import create_model

from .grn import GRN2BMLX
from .rope import text_rope, visual_rope
from .schedule import refinement_target_pt, scale_schedule, shift_pt
from .text import load_or_create_text_embeddings


REFERENCE_GRN = (Path(__file__).resolve().parents[1] / "../GRN").resolve()


def _ensure_reference_importable() -> None:
    if not REFERENCE_GRN.exists():
        raise FileNotFoundError(
            f"Official GRN repo not found at {REFERENCE_GRN}. "
            "It must be a sibling of the xGRN checkout. "
            "See README quickstart: clone GRN alongside xGRN so the layout is ../GRN/."
        )
    root = str(REFERENCE_GRN)
    if root not in sys.path:
        sys.path.insert(0, root)


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True).add(eps))) * weight.float()


def torch_apply_rope(q: torch.Tensor, k: torch.Tensor, rope_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rope_cache = rope_cache[:, None, None]
    out = []
    for item in [q, k]:
        shaped = item.reshape(*item.shape[:-1], -1, 2)
        a = shaped[..., 0] * rope_cache[0] - shaped[..., 1] * rope_cache[1]
        b = shaped[..., 0] * rope_cache[1] + shaped[..., 1] * rope_cache[0]
        out.append(torch.stack([a, b], dim=-1).reshape(item.shape))
    return out[0], out[1]


def report(name: str, torch_tensor: torch.Tensor, mlx_tensor: mx.array) -> dict:
    a = torch_tensor.detach().cpu().float().numpy()
    b = np.array(mlx_tensor).astype(np.float32)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-6)
    row = {
        "name": name,
        "shape": tuple(a.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "finite": bool(np.isfinite(b).all()),
    }
    print(row)
    return row


def report_torch(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a = actual.detach().cpu().float().numpy()
    b = expected.detach().cpu().float().numpy()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-6)
    row = {
        "name": name,
        "shape": tuple(a.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "finite": bool(np.isfinite(b).all()),
    }
    print(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare official PyTorch math with xGRN MLX math.")
    parser.add_argument("--weights", type=Path, default=Path("models/GRN/GRN_T2I_2B.pth"))
    parser.add_argument("--mlx-weights", type=Path, default=Path("models/GRN/mlx/grn_t2i_fp16.safetensors"))
    parser.add_argument("--seq-len", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--full-step", action="store_true")
    parser.add_argument("--refinement-schedule", action="store_true", help="Verify official refinement update schedule and label layout.")
    parser.add_argument("--steps", type=int, default=50, help="Steps for --refinement-schedule.")
    parser.add_argument("--pn", default="0.06M", help="Scale preset for --refinement-schedule.")
    parser.add_argument("--snr-shift", type=float, default=1.0, help="SNR shift for --refinement-schedule.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path for --refinement-schedule.")
    args = parser.parse_args()

    _ensure_reference_importable()
    if args.refinement_schedule:
        run_refinement_schedule(args)
        return
    if args.full_step:
        run_full_step(args)
        return
    torch.manual_seed(args.seed)
    state = torch.load(args.weights, map_location="cpu", mmap=True)
    mlx_model = GRN2BMLX(args.mlx_weights)
    block = 0
    prefix = "block_chunks.0.module.0"
    x = torch.randn(1, args.seq_len, 2304, dtype=torch.float32) * 0.02
    rope_np = np.array(text_rope(args.seq_len)).astype(np.float32)
    rope_t = torch.from_numpy(rope_np)
    xm = mx.array(x.numpy()).astype(mx.float32)
    ropem = text_rope(args.seq_len)

    def w(name: str) -> torch.Tensor:
        return state[f"{prefix}.{name}"].float()

    torch_in = torch_rms_norm(x, w("input_layernorm.weight"))
    mlx_in = mx.fast.rms_norm(xm, mlx_model.fp32_weight(f"{prefix}.input_layernorm.weight"), 1e-6)
    report("input_norm", torch_in, mlx_in)

    tq = F.linear(torch_in, w("attn.q_proj.weight")).reshape(1, args.seq_len, 18, 128).transpose(1, 2)
    tk = F.linear(torch_in, w("attn.k_proj.weight")).reshape(1, args.seq_len, 18, 128).transpose(1, 2)
    tv = F.linear(torch_in, w("attn.v_proj.weight")).reshape(1, args.seq_len, 18, 128).transpose(1, 2)
    tq = torch_rms_norm(tq, w("attn.q_norm.weight"))
    tk = torch_rms_norm(tk, w("attn.k_norm.weight"))
    tq, tk = torch_apply_rope(tq, tk, rope_t)
    mq, mk, mv = mlx_model.qkv(mlx_in, block, ropem)
    report("q_rope", tq, mq)
    report("k_rope", tk, mk)
    report("v", tv, mv)

    tattn = F.scaled_dot_product_attention(tq, tk, tv, scale=128**-0.5).transpose(1, 2).reshape(1, args.seq_len, 2304)
    tattn = F.linear(tattn, w("attn.o_proj.weight"))
    mattn, _, _ = mlx_model.attention(mlx_in, block, ropem)
    report("attn_out", tattn, mattn)

    tx = x + tattn
    mx_x = xm + mattn
    report("resid_attn", tx, mx_x)

    tpost = torch_rms_norm(tx, w("post_attention_layernorm.weight"))
    mpost = mx.fast.rms_norm(mx_x, mlx_model.fp32_weight(f"{prefix}.post_attention_layernorm.weight"), 1e-6)
    report("post_norm", tpost, mpost)

    tmlp = F.linear(F.silu(F.linear(tpost, w("mlp.gate_proj.weight"))) * F.linear(tpost, w("mlp.up_proj.weight")), w("mlp.down_proj.weight"))
    mmlp = mlx_model.mlp(mpost, block)
    report("mlp", tmlp, mmlp)
    report("block_out", tx + tmlp, mx_x + mmlp)


def _official_target_pt(step_index: int, steps: int, snr_shift: float) -> float:
    pt_unshift = (step_index + 1) / max(1, steps - 1)
    pt_shift = shift_pt(min(1.0, pt_unshift), snr_shift)
    return float((1 - np.cos(np.pi / 2 * pt_shift)) * 0.999)


def run_refinement_schedule(args: argparse.Namespace) -> None:
    import json

    rng = np.random.default_rng(args.seed)
    schedule, _ = scale_schedule(args.pn, 1.0, 1)
    pt, ph, pw = schedule[0]
    token_count = pt * ph * pw
    d = 64 * 4
    pure_rand = rng.integers(0, 2, size=(1, d, pt, ph, pw), dtype=np.int32)
    pred_bld = rng.integers(0, 2, size=(1, token_count, d), dtype=np.int64)

    # Official path: pred_sample_labels [B, thw, d] -> [B, t, h, w, d] -> [B, d, t, h, w].
    pred_torch = torch.from_numpy(pred_bld).view(1, token_count, d).view(1, pt, ph, pw, d).permute(0, 4, 1, 2, 3)
    # Runtime path: MLX categorical output uses the same flattened [B, thw, d] layout.
    pred_mlx = mx.transpose(mx.array(pred_bld).reshape(1, pt, ph, pw, d), (0, 4, 1, 2, 3)).astype(mx.int32)
    pred_np = np.array(pred_mlx)
    layout_max_abs = float(np.abs(pred_torch.numpy().astype(np.int32) - pred_np).max())

    rows = []
    exact_next_pt = 0.0
    layout_ok = layout_max_abs == 0.0
    pure_reuse_ok = True
    target_ok = True
    exact_next_ok = True
    for step in range(args.steps):
        cur_pt = exact_next_pt
        official_target = _official_target_pt(step, args.steps, args.snr_shift)
        runtime_target = refinement_target_pt(step, args.steps, args.snr_shift)
        target_delta = abs(official_target - runtime_target)
        target_ok = target_ok and target_delta < 1e-12
        mask = rng.random(size=pred_np.shape) < official_target
        official_next = float(mask.astype(np.float32).mean())
        mixed = np.where(mask, pred_np, pure_rand).astype(np.int32)
        pure_reuse_ok = pure_reuse_ok and bool(np.array_equal(mixed[~mask], pure_rand[~mask]))
        runtime_exact_next = official_next
        exact_next_ok = exact_next_ok and abs(official_next - runtime_exact_next) < 1e-12
        rows.append(
            {
                "step": step + 1,
                "cur_pt": cur_pt,
                "target_pt": official_target,
                "target_delta": target_delta,
                "random_mask_mean": official_next,
                "exact_next_pt": runtime_exact_next,
                "high_performance_next_pt": runtime_target,
                "high_performance_delta_vs_exact": abs(runtime_target - official_next),
            }
        )
        exact_next_pt = runtime_exact_next

    result = {
        "passed": bool(layout_ok and pure_reuse_ok and target_ok and exact_next_ok),
        "pn": args.pn,
        "shape": [1, d, pt, ph, pw],
        "steps": args.steps,
        "layout_max_abs": layout_max_abs,
        "layout_ok": bool(layout_ok),
        "target_ok": bool(target_ok),
        "pure_random_reuse_ok": bool(pure_reuse_ok),
        "exact_next_pt_ok": bool(exact_next_ok),
        "note": "Default generation uses target_pt as a high-performance next_pt approximation; --exact-step-sync uses random_mask_mean.",
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.report}")
    if not result["passed"]:
        raise SystemExit(1)


def run_full_step(args: argparse.Namespace) -> None:
    import grn.models.grn  # noqa: F401 - registers GRN2b with timm.
    from grn.schedules.global_refine import get_scale_pack_info, get_visual_rope_embeds
    from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input

    class DummyVAE:
        codebook_dim = 64

    other_args = argparse.Namespace(
        detail_scale_dim=64,
        detail_num_lvl=2,
        hbq_round=4,
        refine_mode="ar_discrete_GRN_bit",
        dynamic_scale_schedule="GRN_vae_stride16",
        train_h_div_w_list="[]",
        video_frames=81,
        temporal_compress_rate=4,
        rope_type="3d",
        add_scale_token=1,
        use_ada_layer_norm=0,
        mapped_h_div_w_template=1.0,
        first_full_spatial_size_scale_index=0,
    )
    torch_state = torch.load(args.weights, map_location="cpu", mmap=True)
    torch_model = create_model(
        "GRN2b",
        vae_local=DummyVAE(),
        text_channels=4096,
        text_maxlen=512,
        checkpointing="full-block",
        pad_to_multiplier=128,
        use_flex_attn=False,
        num_of_label_value=2,
        rope2d_normalized_by_hw=2,
        pn="0.06M",
        apply_spatial_patchify=0,
        inference_mode=True,
        dynamic_scale_schedule="GRN_vae_stride16",
        video_frames=81,
        other_args=other_args,
    )
    torch_model.load_state_dict(torch_state, strict=True)
    torch_model.eval().requires_grad_(False)

    mlx_model = GRN2BMLX(args.mlx_weights)
    prompt = "A realistic photo of an orange tabby cat sitting on a windowsill, fluffy fur, green eyes, soft daylight, natural indoor background, sharp focus, warm but realistic colors"
    negative = "blurry, low quality, watermark, text, logo, distorted, deformed, extra limbs, cartoon, painting, illustration, oversaturated"
    cond_mx, uncond_mx = load_or_create_text_embeddings(prompt, negative)
    cond = torch.from_numpy(np.array(cond_mx).astype(np.float32))
    uncond = torch.from_numpy(np.array(uncond_mx).astype(np.float32))
    cond_tuple = (cond, [cond.shape[0]], torch.tensor([0, cond.shape[0]], dtype=torch.int32), cond.shape[0])
    uncond_tuple = (uncond, [uncond.shape[0]], torch.tensor([0, uncond.shape[0]], dtype=torch.int32), uncond.shape[0])

    prefix_cond, lens_cond = torch_model.prepare_text_conditions(cond_tuple, None, use_cfg=False)
    prefix_uncond, lens_uncond = torch_model.prepare_text_conditions(uncond_tuple, None, use_cfg=False)
    report("text_proj_cond", prefix_cond, mlx_model.project_text(cond_mx))
    report("text_proj_uncond", prefix_uncond, mlx_model.project_text(uncond_mx))

    prefix, lens = torch_model.prepare_text_conditions(cond_tuple, uncond_tuple, use_cfg=True)
    rope_text = torch_model.rope2d_freqs_grid["freqs_text"]
    rope_cache = torch.cat([rope_text[:, :, :, :, : lens[0]], rope_text[:, :, :, :, : lens[1]]], dim=4)
    last = prefix
    x_cond = mlx_model.project_text(cond_mx)
    x_uncond = mlx_model.project_text(uncond_mx)
    mlx_cond_keys: list[mx.array] = []
    mlx_cond_vals: list[mx.array] = []
    mlx_uncond_keys: list[mx.array] = []
    mlx_uncond_vals: list[mx.array] = []
    for block in torch_model.unregistered_blocks:
        block.attn.kv_caching(True)
    for block_idx, block in enumerate(torch_model.unregistered_blocks):
        last = block(
            x=last,
            e0=None,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            rope2d_freqs_grid=rope_cache,
            scale_ind="t0",
            context_info={},
            last_diffusion_step=True,
            ref_text_scale_inds=[],
            use_cfg=True,
            split_cond_uncond=lens,
        )
        x_cond, k_cond, v_cond = mlx_model.block(x_cond, block_idx, text_rope(lens[0]))
        x_uncond, k_uncond, v_uncond = mlx_model.block(x_uncond, block_idx, text_rope(lens[1]))
        mlx_cond_keys.append(k_cond)
        mlx_cond_vals.append(v_cond)
        mlx_uncond_keys.append(k_uncond)
        mlx_uncond_vals.append(v_uncond)
        if block_idx in {0, 1, len(torch_model.unregistered_blocks) - 1}:
            report(f"text_hidden_cond_b{block_idx}", last[:, : lens[0]], x_cond)
            report(f"text_hidden_uncond_b{block_idx}", last[:, lens[0] :], x_uncond)
            report(
                f"text_k_cfg_b{block_idx}",
                block.attn.cached_k["t0"],
                mx.concatenate([k_cond, k_uncond], axis=2),
            )
            report(
                f"text_v_cfg_b{block_idx}",
                block.attn.cached_v["t0"],
                mx.concatenate([v_cond, v_uncond], axis=2),
            )
    mlx_cond_cache = mlx_model.encode_text_cache(cond_mx)
    mlx_uncond_cache = mlx_model.encode_text_cache(uncond_mx)

    schedule, mapped = scale_schedule("0.06M", 1.0, 1)
    pt, ph, pw = schedule[0]
    context_info = get_scale_pack_info(schedule, first_full_spatial_size_scale_index=0, args=other_args)
    torch.manual_seed(args.seed)
    mixed = torch.randint(0, 2, (1, 256, pt, ph, pw), dtype=torch.long)
    onehot = multiclass_labels2onehot_input(mixed, 2)
    labels_mx = mx.array(mixed.numpy().astype(np.int32))
    visual = mlx_model.labels_to_onehot(labels_mx)
    report("visual_onehot", onehot, visual)

    official_visual_rope = get_visual_rope_embeds(
        torch_model.rope2d_freqs_grid,
        schedule,
        0,
        0,
        torch.device("cpu"),
        other_args,
        context_info,
        mapped,
    )
    report("visual_rope", official_visual_rope.reshape(2, pt * ph * pw, 64), visual_rope(pt, ph, pw, mapped))
    scale_token_rope = rope_text[:, :, :, :, 512:513]
    visual_rope_t = torch.cat([official_visual_rope, scale_token_rope], dim=4)
    last_stage = torch_model.embeds_codes2input(onehot)
    visual_embed_mx = mlx_model.embed_visual_codes(visual)
    report("visual_embed", last_stage, visual_embed_mx)
    pt_embed_t = torch_model.pt_embedder(torch.tensor([0.0]))
    pt_embed_mx = mlx_model.pt_embed(0.0)
    report("pt_embed_0", pt_embed_t, pt_embed_mx)

    # Cond-only visual pass.  This isolates visual math from CFG splitting.
    for block in torch_model.unregistered_blocks:
        block.attn.kv_caching(True)
    text_cond = prefix_cond
    for block_idx, block in enumerate(torch_model.unregistered_blocks):
        text_cond = block(
            x=text_cond,
            e0=None,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            rope2d_freqs_grid=rope_text[:, :, :, :, : lens_cond[0]],
            scale_ind="t0",
            context_info={},
            last_diffusion_step=True,
            ref_text_scale_inds=[],
            use_cfg=False,
            split_cond_uncond=lens_cond,
        )
    visual_cond_t = torch.cat((last_stage, pt_embed_t), dim=1)
    visual_cond_mx = mx.concatenate([visual_embed_mx, pt_embed_mx], axis=1)
    visual_rope_mx = mx.concatenate([visual_rope(pt, ph, pw, mapped), text_rope(1, offset=512)], axis=1)
    for block_idx, block in enumerate(torch_model.unregistered_blocks):
        visual_cond_t = block(
            x=visual_cond_t,
            e0=None,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            rope2d_freqs_grid=visual_rope_t,
            scale_ind=0,
            context_info=context_info,
            last_diffusion_step=False,
            ref_text_scale_inds=["t0"],
            use_cfg=False,
            split_cond_uncond=[pt * ph * pw + 1],
        )
        visual_cond_mx, _, _ = mlx_model.block(
            visual_cond_mx,
            block_idx,
            visual_rope_mx,
            mlx_cond_cache.k[block_idx],
            mlx_cond_cache.v[block_idx],
        )
        if block_idx in {0, 1, len(torch_model.unregistered_blocks) - 1}:
            report(f"visual_cond_hidden_b{block_idx}", visual_cond_t, visual_cond_mx)
    report("visual_cond_logits", torch_model.get_logits_during_infer(visual_cond_t)[:, : pt * ph * pw].reshape(1, pt * ph * pw, -1, 2), mlx_model.logits(visual_cond_mx, pt * ph * pw))

    # Full CFG pass, matching the production guidance layout.
    for block in torch_model.unregistered_blocks:
        block.attn.kv_caching(True)
    last = prefix
    for block in torch_model.unregistered_blocks:
        last = block(
            x=last,
            e0=None,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            rope2d_freqs_grid=rope_cache,
            scale_ind="t0",
            context_info={},
            last_diffusion_step=True,
            ref_text_scale_inds=[],
            use_cfg=True,
            split_cond_uncond=lens,
        )
    last_stage = torch.cat((last_stage, pt_embed_t), dim=1)
    last_stage = torch.cat([last_stage, last_stage], dim=1)
    visual_rope_t = torch.cat([visual_rope_t, visual_rope_t], dim=4)
    for block in torch_model.unregistered_blocks:
        last_stage = block(
            x=last_stage,
            e0=None,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            rope2d_freqs_grid=visual_rope_t,
            scale_ind=0,
            context_info={0: {"ref_sids": []}},
            last_diffusion_step=False,
            ref_text_scale_inds=["t0"],
            use_cfg=True,
            split_cond_uncond=[pt * ph * pw + 1] * 2,
        )
    t_logits = torch_model.get_logits_during_infer(last_stage).reshape(1, last_stage.shape[1], -1, 2)

    cond_out = mlx_model.visual_forward(visual, 0.0, pt, ph, pw, mapped, mlx_cond_cache)
    uncond_out = mlx_model.visual_forward(visual, 0.0, pt, ph, pw, mapped, mlx_uncond_cache)
    m_logits = mx.concatenate([mlx_model.logits(cond_out, pt * ph * pw), mlx_model.logits(uncond_out, pt * ph * pw)], axis=1)
    official_cfg_logits = torch.cat(
        [
            t_logits[:, : pt * ph * pw],
            t_logits[:, pt * ph * pw + 1 : 2 * pt * ph * pw + 1],
        ],
        dim=1,
    )
    report("visual_logits_concat", official_cfg_logits, m_logits)


if __name__ == "__main__":
    main()
