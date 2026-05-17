"""Test whether q/k/v are actually serial inside the compiled visual_pass.

If they were perfectly parallel, running (q,k,v) together should cost
≈ max(t_q, t_k, t_v). If serial, it costs ≈ sum.

We can't directly probe inside mx.compile, but we CAN microbench three
parallel-looking matmuls vs three sequenced ones with mx.eval forcing
in-order execution. The ratio tells us how much overlap MLX/Metal
actually achieves.
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

    # warm
    for _ in range(20):
        mx.eval(x @ w_q, x @ w_k, x @ w_v)

    N_iter = 200

    # A) single matmul time
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_q)
    t_single = (time.perf_counter() - t0) / N_iter * 1e6

    # B) three matmuls in one eval call (MLX could in theory overlap)
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_q, x @ w_k, x @ w_v)
    t_three_one_eval = (time.perf_counter() - t0) / N_iter * 1e6

    # C) three matmuls with sync between each (forced serial)
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_q)
        mx.eval(x @ w_k)
        mx.eval(x @ w_v)
    t_three_separate_eval = (time.perf_counter() - t0) / N_iter * 1e6

    # D) compile the three together (mimics what mx.compile does in qkv())
    @mx.compile
    def qkv(x_, wq_, wk_, wv_):
        return x_ @ wq_, x_ @ wk_, x_ @ wv_

    for _ in range(20):
        q, k, v = qkv(x, w_q, w_k, w_v); mx.eval(q, k, v)

    t0 = time.perf_counter()
    for _ in range(N_iter):
        q, k, v = qkv(x, w_q, w_k, w_v); mx.eval(q, k, v)
    t_compiled = (time.perf_counter() - t0) / N_iter * 1e6

    print(f"shape  M={M}, K={K}, N={N}  bf16")
    print()
    print(f"A) 1 matmul (q only):                          {t_single:7.1f} us")
    print(f"B) 3 matmuls, single mx.eval (lazy queue):     {t_three_one_eval:7.1f} us")
    print(f"C) 3 matmuls, forced serial (eval each):       {t_three_separate_eval:7.1f} us")
    print(f"D) mx.compile'd (q,k,v) -> tuple:              {t_compiled:7.1f} us")
    print()
    print(f"Ratio B/A (vs single matmul): {t_three_one_eval/t_single:.2f}× — "
          f"{'≈3× ⇒ SERIAL on Metal queue' if t_three_one_eval/t_single >= 2.5 else 'closer to 1× ⇒ overlapped'}")
    print(f"Ratio D/A (compiled vs 1):    {t_compiled/t_single:.2f}× — what production qkv() looks like")


if __name__ == "__main__":
    main()
