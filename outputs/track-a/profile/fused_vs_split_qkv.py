"""Compare fused vs split QKV matmul.

If we concatenate q/k/v weights into a single [K, 3N] tensor and do ONE matmul,
does it beat 3 separate matmuls?

Hypothesis: a single larger matmul amortizes Metal kernel launch overhead and
may achieve higher % of peak FLOPs.  We've previously seen --fuse-qkv-concat
regress inside mx.compile (+4.75%); test it cleanly outside compile here.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np


def main():
    M, K, N = 514, 2304, 2304
    rng = np.random.RandomState(0)
    x = mx.array(rng.randn(M, K).astype(np.float32) * 0.1).astype(mx.bfloat16)
    w_q = mx.array(rng.randn(K, N).astype(np.float32) * 0.05).astype(mx.bfloat16)
    w_k = mx.array(rng.randn(K, N).astype(np.float32) * 0.05).astype(mx.bfloat16)
    w_v = mx.array(rng.randn(K, N).astype(np.float32) * 0.05).astype(mx.bfloat16)
    w_fused = mx.concatenate([w_q, w_k, w_v], axis=1)  # [K, 3N]
    mx.eval(x, w_q, w_k, w_v, w_fused)

    N_iter = 300

    # warmups
    for _ in range(20):
        mx.eval(x @ w_q, x @ w_k, x @ w_v)
        mx.eval(x @ w_fused)

    # A) 3 separate matmuls
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_q, x @ w_k, x @ w_v)
    t_split = (time.perf_counter() - t0) / N_iter * 1e6

    # B) 1 fused matmul (output [M, 3N])
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_fused)
    t_fused = (time.perf_counter() - t0) / N_iter * 1e6

    # C) 1 fused matmul + manual split (slice into views — should be free)
    t0 = time.perf_counter()
    for _ in range(N_iter):
        qkv = x @ w_fused
        q, k, v = mx.split(qkv, 3, axis=1)
        mx.eval(q, k, v)
    t_fused_split = (time.perf_counter() - t0) / N_iter * 1e6

    # D) compile the fused-split path
    @mx.compile
    def qkv_fused_compiled(x_, w_):
        qkv = x_ @ w_
        return mx.split(qkv, 3, axis=1)

    for _ in range(20):
        q, k, v = qkv_fused_compiled(x, w_fused)
        mx.eval(q, k, v)

    t0 = time.perf_counter()
    for _ in range(N_iter):
        q, k, v = qkv_fused_compiled(x, w_fused)
        mx.eval(q, k, v)
    t_fused_compiled = (time.perf_counter() - t0) / N_iter * 1e6

    # E) compile the split path (mimics current production)
    @mx.compile
    def qkv_split_compiled(x_, wq_, wk_, wv_):
        return x_ @ wq_, x_ @ wk_, x_ @ wv_

    for _ in range(20):
        q, k, v = qkv_split_compiled(x, w_q, w_k, w_v)
        mx.eval(q, k, v)

    t0 = time.perf_counter()
    for _ in range(N_iter):
        q, k, v = qkv_split_compiled(x, w_q, w_k, w_v)
        mx.eval(q, k, v)
    t_split_compiled = (time.perf_counter() - t0) / N_iter * 1e6

    print(f"shape M={M} K={K} N={N}  bf16  iters={N_iter}")
    print()
    print(f"A) split   · 3 matmul · eager:       {t_split:7.1f} us")
    print(f"B) fused   · 1 matmul · eager · raw: {t_fused:7.1f} us  ({t_fused/t_split*100:.1f}% of A)")
    print(f"C) fused   · 1 matmul + split:       {t_fused_split:7.1f} us  ({t_fused_split/t_split*100:.1f}% of A)")
    print(f"D) fused   · compiled:               {t_fused_compiled:7.1f} us  ({t_fused_compiled/t_split*100:.1f}% of A)")
    print(f"E) split   · compiled (production):  {t_split_compiled:7.1f} us  ({t_split_compiled/t_split*100:.1f}% of A)")
    print()
    print(f"Best vs production (E):")
    candidates = {"A": t_split, "B": t_fused, "C": t_fused_split, "D": t_fused_compiled}
    best_name = min(candidates, key=lambda k: candidates[k])
    best = candidates[best_name]
    delta = (best / t_split_compiled - 1) * 100
    print(f"  best = {best_name} = {best:.1f} us  ·  vs E ({t_split_compiled:.1f}) → {delta:+.1f}%")
    if delta < -3:
        print(f"  ★ a fused variant beats production — worth integrating")
    else:
        print(f"  ✗ no meaningful win — split-compiled is already optimal")


if __name__ == "__main__":
    main()
