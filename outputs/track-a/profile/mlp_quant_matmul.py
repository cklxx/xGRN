"""Benchmark MLX quantized matmul on the three MLP shapes.

MLP shapes (per block, with B=2 CFG batched, L=257):
  gate_proj  [514, 2304] @ [2304, 8192]    weight 37.7 MB bf16
  up_proj    [514, 2304] @ [2304, 8192]    weight 37.7 MB bf16
  down_proj  [514, 8192] @ [8192, 2304]    weight 37.7 MB bf16

Working set per call ≈ 51-57 MB → 4-4.5× L2 (12 MB) → bandwidth-streaming.

mx.quantize bits options: 2, 3, 4, 6, 8.
group_size options: 32, 64, 128 (default 64).

Compression:
  bf16          → 2.00 B / weight   (baseline)
  int8 gs=128   → 1.04 B / weight   ~52%
  int8 gs=64    → 1.08 B / weight
  int4 gs=128   → 0.54 B / weight   ~27%
  int4 gs=64    → 0.58 B / weight
  int4 gs=32    → 0.62 B / weight

If MLP is BW-bound, savings should track compression ratio.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np


SHAPES = [
    ("gate_proj", 514, 2304, 8192),
    ("up_proj",   514, 2304, 8192),
    ("down_proj", 514, 8192, 2304),
]

CONFIGS = [
    ("bf16",     None, None),
    ("int8 gs=128", 8, 128),
    ("int8 gs=64",  8,  64),
    ("int4 gs=128", 4, 128),
    ("int4 gs=64",  4,  64),
    ("int4 gs=32",  4,  32),
]


def bench_one(M, K, N, bits, group_size, N_iter=200):
    rng = np.random.RandomState(0)
    x = mx.array(rng.randn(M, K).astype(np.float32) * 0.1).astype(mx.bfloat16)
    w = mx.array(rng.randn(N, K).astype(np.float32) * 0.05).astype(mx.bfloat16)
    # MLX convention: quantized_matmul expects weights in [N, K] shape (output dim first)
    mx.eval(x, w)

    if bits is None:
        # bf16 baseline: x @ w.T to produce [M, N]
        # warmup
        for _ in range(20):
            mx.eval(x @ w.T)
        t0 = time.perf_counter()
        for _ in range(N_iter):
            mx.eval(x @ w.T)
        return (time.perf_counter() - t0) / N_iter * 1e6
    else:
        w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(w_q, scales, biases)
        # warmup
        for _ in range(20):
            mx.eval(mx.quantized_matmul(x, w_q, scales=scales, biases=biases,
                                         transpose=True, group_size=group_size, bits=bits))
        t0 = time.perf_counter()
        for _ in range(N_iter):
            mx.eval(mx.quantized_matmul(x, w_q, scales=scales, biases=biases,
                                         transpose=True, group_size=group_size, bits=bits))
        return (time.perf_counter() - t0) / N_iter * 1e6


def main():
    print(f"{'op':<12} {'config':<14} {'us':>8} {'vs bf16':>10}")
    print("-" * 50)
    for name, M, K, N in SHAPES:
        baseline = None
        for cfg_name, bits, gs in CONFIGS:
            try:
                us = bench_one(M, K, N, bits, gs)
                if baseline is None:
                    baseline = us
                ratio = f"{us/baseline*100:.1f}%"
            except Exception as e:
                us = float('nan')
                ratio = f"FAIL: {str(e)[:30]}"
            print(f"{name:<12} {cfg_name:<14} {us:8.1f} {ratio:>10}")
        print()


if __name__ == "__main__":
    main()
