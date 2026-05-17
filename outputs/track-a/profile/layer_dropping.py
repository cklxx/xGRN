"""P1-2 Layer dropping spike: drop block N, check wall + CLIP via reuse of xgrn validate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from xgrn_mlx.grn import GRN2BMLX
from xgrn_mlx.text import load_or_create_text_embeddings_with_key


PROMPT = (
    "A realistic photo of an orange tabby cat sitting on a windowsill, "
    "fluffy fur, green eyes, soft daylight, natural indoor background"
)
NEG = ""


def measure(drop_blocks: tuple[int, ...], N: int = 2) -> tuple[float, mx.array]:
    mx.clear_cache()
    m = GRN2BMLX(
        "models/GRN/mlx_fp32/grn_t2i_fp32.safetensors",
        compute_dtype="bf16",
        compile_visual_pass=True,
        drop_blocks=drop_blocks,
    )
    cond, uncond, key = load_or_create_text_embeddings_with_key(
        PROMPT, NEG, model_dir=Path("models/GRN"), text_dtype="bf16", cache_dtype="fp32",
    )
    # warm
    raw, _ = m.refine(
        cond, uncond, pt=1, ph=16, pw=16, mapped_h_div_w=1.0, steps=50,
        guidance=3.0, temperature=1.1, seed=42,
        cond_cache_key=f"{key}:cond", uncond_cache_key=f"{key}:uncond",
    )
    mx.eval(raw)
    times = []
    last_raw = None
    for _ in range(N):
        t0 = time.perf_counter()
        raw, _ = m.refine(
            cond, uncond, pt=1, ph=16, pw=16, mapped_h_div_w=1.0, steps=50,
            guidance=3.0, temperature=1.1, seed=42,
            cond_cache_key=f"{key}:cond", uncond_cache_key=f"{key}:uncond",
        )
        mx.eval(raw)
        times.append(time.perf_counter() - t0)
        last_raw = raw
    m.close()
    return float(np.median(times)), last_raw


def main():
    configs = [
        (),
        (14,),
        (13, 14),
        (14, 15),
        (10, 14, 20),
    ]
    base_t = None
    results = {}
    for cfg in configs:
        t, _ = measure(cfg)
        if base_t is None:
            base_t = t
        delta = t - base_t
        pct = delta / base_t * 100
        label = f"drop={cfg}" if cfg else "baseline"
        print(f"  {label:<32s}  refine {t:.2f}s  Δ {delta:+.2f}s ({pct:+.1f}%)")
        results[str(cfg)] = {"refine_s": t, "delta_s": delta, "delta_pct": pct}
    Path("outputs/track-a/profile/layer_dropping.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
