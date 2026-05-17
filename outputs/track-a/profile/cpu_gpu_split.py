"""Quick CPU vs GPU breakdown.

On Apple Silicon, CPU and GPU share physical memory — there is no DMA
between them. What we can still measure cheaply:

  - Wall time without forcing eval: CPU-side cost of building the
    MLX graph + queueing Metal command buffers. The GPU work has not
    happened yet at this point (lazy eval).
  - Wall time with mx.eval at the end: total time including GPU work
    + the sync barrier (CPU waits for GPU to finish).

The difference is GPU-bound time (modulo CPU/GPU overlap from
the next-iteration dispatch). We instrument a single refinement step
in this script so the numbers are directly comparable to the phase
profile.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from xgrn_mlx.grn import GRN2BMLX
from xgrn_mlx.rope import text_rope, visual_rope
from xgrn_mlx.run import _model
from xgrn_mlx.schedule import refinement_target_pt, scale_schedule
from xgrn_mlx.text import load_or_create_text_embeddings_with_key


def main():
    prompt = (
        "A realistic photo of an orange tabby cat sitting on a windowsill, "
        "fluffy fur, green eyes, soft daylight, natural indoor background"
    )
    cond, uncond, key = load_or_create_text_embeddings_with_key(
        prompt, "", model_dir=Path("models/GRN"),
        text_dtype="bf16", cache_dtype="fp32",
    )
    schedule, mapped = scale_schedule("0.25M", 1.0, 1)
    pt, ph, pw = schedule[0]
    model: GRN2BMLX = _model(
        "t2i", Path("models/GRN"),
        weights_dtype="fp32", compute_dtype="bf16",
        compile_visual_pass=True,
    )
    cond_cache = model.encode_text_cache_cached(cond, f"{key}:cond")
    uncond_cache = model.encode_text_cache_cached(uncond, f"{key}:uncond")
    d = model.config.vae_latent_dim * model.config.hbq_round
    pure_rand = mx.random.randint(0, model.config.detail_num_lvl, (1, d, pt, ph, pw), dtype=mx.int32)
    rope = mx.concatenate([visual_rope(pt, ph, pw, mapped), text_rope(1, offset=512)], axis=1)
    mx.eval(rope)
    cfg_cache = model.make_cfg_cache(cond_cache, uncond_cache, pt * ph * pw + 1)
    token_count = pt * ph * pw
    mixed = pure_rand

    # warm
    for _ in range(3):
        pt_embed_token = model.pt_embed(0.0)
        visual_input = mx.concatenate(
            [model.embed_visual_labels(mixed), pt_embed_token], axis=1
        )
        out = model.visual_forward_cfg_batched(visual_input, rope, cfg_cache)
        cfg_logits = model.logits(out, token_count)
        logits = (cfg_logits[1:2] + 3.0 * (cfg_logits[:1] - cfg_logits[1:2])) / 1.1
        sample = mx.random.categorical(logits.reshape(-1, 2)).reshape(1, token_count, d)
        pred = mx.transpose(sample.reshape(1, pt, ph, pw, d), (0, 4, 1, 2, 3)).astype(mx.int32)
        mask = mx.random.uniform(shape=pred.shape) < 0.5
        mixed = mx.where(mask, pred, pure_rand).astype(mx.int32)
        mx.eval(mixed)

    N = 30
    # ---- Measurement A: pure dispatch time (lazy queueing, no mx.eval) ----
    dispatch_times = []
    final_outs = []
    for _ in range(N):
        t0 = time.perf_counter()
        pt_embed_token = model.pt_embed(0.0)
        visual_input = mx.concatenate(
            [model.embed_visual_labels(mixed), pt_embed_token], axis=1
        )
        out = model.visual_forward_cfg_batched(visual_input, rope, cfg_cache)
        cfg_logits = model.logits(out, token_count)
        logits = (cfg_logits[1:2] + 3.0 * (cfg_logits[:1] - cfg_logits[1:2])) / 1.1
        sample = mx.random.categorical(logits.reshape(-1, 2)).reshape(1, token_count, d)
        pred = mx.transpose(sample.reshape(1, pt, ph, pw, d), (0, 4, 1, 2, 3)).astype(mx.int32)
        mask = mx.random.uniform(shape=pred.shape) < 0.5
        mixed_next = mx.where(mask, pred, pure_rand).astype(mx.int32)
        # No mx.eval here — just queue everything.
        dispatch_times.append(time.perf_counter() - t0)
        final_outs.append(mixed_next)

    # Drain the queue
    mx.eval(*final_outs)
    cpu_dispatch_ms = float(np.median(dispatch_times) * 1000)

    # ---- Measurement B: full wall (queue + force eval) ----
    full_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        pt_embed_token = model.pt_embed(0.0)
        visual_input = mx.concatenate(
            [model.embed_visual_labels(mixed), pt_embed_token], axis=1
        )
        out = model.visual_forward_cfg_batched(visual_input, rope, cfg_cache)
        cfg_logits = model.logits(out, token_count)
        logits = (cfg_logits[1:2] + 3.0 * (cfg_logits[:1] - cfg_logits[1:2])) / 1.1
        sample = mx.random.categorical(logits.reshape(-1, 2)).reshape(1, token_count, d)
        pred = mx.transpose(sample.reshape(1, pt, ph, pw, d), (0, 4, 1, 2, 3)).astype(mx.int32)
        mask = mx.random.uniform(shape=pred.shape) < 0.5
        mixed_next = mx.where(mask, pred, pure_rand).astype(mx.int32)
        mx.eval(mixed_next)
        full_times.append(time.perf_counter() - t0)
        mixed = mixed_next

    wall_ms = float(np.median(full_times) * 1000)
    # GPU-attributable time = wall − CPU dispatch (under-estimate; some
    # CPU/GPU overlap means the real GPU work may be slightly longer)
    gpu_attributable_ms = wall_ms - cpu_dispatch_ms

    out_path = Path("outputs/track-a/profile/cpu_gpu_split.json")
    out_path.write_text(json.dumps({
        "cpu_dispatch_ms": cpu_dispatch_ms,
        "wall_ms": wall_ms,
        "gpu_attributable_ms": gpu_attributable_ms,
        "n_samples": N,
        "notes": (
            "CPU dispatch = time to queue all MLX ops without forcing eval "
            "(Python loop + MLX op build + Metal encoder calls). "
            "Wall = same with mx.eval at the end (CPU waits for GPU). "
            "GPU-attributable = wall − CPU dispatch (under-estimate due to "
            "CPU/GPU overlap; in steady state most of wall is real GPU work)."
        ),
    }, indent=2))

    print(f"CPU dispatch (lazy queue):  {cpu_dispatch_ms:7.2f} ms / step")
    print(f"Wall (queue + eval):        {wall_ms:7.2f} ms / step")
    print(f"GPU-attributable:           {gpu_attributable_ms:7.2f} ms / step "
          f"({gpu_attributable_ms/wall_ms*100:.1f} %)")
    print(f"CPU share:                  {cpu_dispatch_ms/wall_ms*100:.1f} %")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
