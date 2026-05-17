"""Test whether putting q/k/v on separate mx.Stream's actually parallelizes
on Metal.

Background — from outputs/track-a/profile/serial_vs_parallel.py:
  1 matmul:                        1061 us
  3 matmul (single mx.eval):       2663 us  → 2.51×
  3 matmul (forced eval each):     3195 us  → 3.01×
  mx.compile (q,k,v) tuple:        2682 us  → 2.53×

i.e. MLX's default behavior gives us ~16% overlap.  The hypothesis:
mx.new_stream(mx.gpu) gives each op its own Metal MTLCommandQueue,
which Apple's GPU scheduler can run concurrently across the 20 cores.
A single matmul here uses ~2.5 TFLOP/s (42% of M4 Pro peak ~6 TFLOP/s),
so there are unused cores to harvest.

This script measures:
  E) 3 matmuls, each on its own GPU stream  · single mx.eval
  F) 3 matmuls, each on its own GPU stream  · forced ordering (eval each)
  G) baseline 1 matmul · explicit gpu stream
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
    mx.eval(x, w_q, w_k, w_v)

    s_q = mx.new_stream(mx.gpu)
    s_k = mx.new_stream(mx.gpu)
    s_v = mx.new_stream(mx.gpu)
    default = mx.default_stream(mx.gpu)

    print(f"streams:")
    print(f"  default: {default}")
    print(f"  s_q:     {s_q}")
    print(f"  s_k:     {s_k}")
    print(f"  s_v:     {s_v}")
    print()

    # --- warmups for everything we'll measure ---
    for _ in range(20):
        mx.eval(x @ w_q)

    for _ in range(20):
        mx.eval(x @ w_q, x @ w_k, x @ w_v)

    for _ in range(20):
        q = mx.matmul(x, w_q, stream=s_q)
        k = mx.matmul(x, w_k, stream=s_k)
        v = mx.matmul(x, w_v, stream=s_v)
        mx.eval(q, k, v)

    N_iter = 300

    # --- A) baseline: 1 matmul default stream ---
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_q)
    t_single = (time.perf_counter() - t0) / N_iter * 1e6

    # --- B) 3 matmuls, single mx.eval, default stream ---
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(x @ w_q, x @ w_k, x @ w_v)
    t_three_default = (time.perf_counter() - t0) / N_iter * 1e6

    # --- C) compiled tuple, default stream ---
    @mx.compile
    def qkv_default(x_, wq_, wk_, wv_):
        return x_ @ wq_, x_ @ wk_, x_ @ wv_

    for _ in range(20):
        q, k, v = qkv_default(x, w_q, w_k, w_v)
        mx.eval(q, k, v)

    t0 = time.perf_counter()
    for _ in range(N_iter):
        q, k, v = qkv_default(x, w_q, w_k, w_v)
        mx.eval(q, k, v)
    t_compiled = (time.perf_counter() - t0) / N_iter * 1e6

    # --- E) 3 matmuls on 3 separate streams ---
    t0 = time.perf_counter()
    for _ in range(N_iter):
        q = mx.matmul(x, w_q, stream=s_q)
        k = mx.matmul(x, w_k, stream=s_k)
        v = mx.matmul(x, w_v, stream=s_v)
        mx.eval(q, k, v)
    t_three_streams = (time.perf_counter() - t0) / N_iter * 1e6

    # --- F) 3 matmuls compiled, with explicit per-op stream ---
    # mx.compile doesn't expose per-op stream so we test the un-compiled version
    # with explicit eval ordering.
    t0 = time.perf_counter()
    for _ in range(N_iter):
        q = mx.matmul(x, w_q, stream=s_q)
        k = mx.matmul(x, w_k, stream=s_k)
        v = mx.matmul(x, w_v, stream=s_v)
        mx.eval(q)
        mx.eval(k)
        mx.eval(v)
    t_three_streams_eachsync = (time.perf_counter() - t0) / N_iter * 1e6

    # --- G) all 3 on the SAME new stream (sanity: should be ~= B) ---
    t0 = time.perf_counter()
    for _ in range(N_iter):
        q = mx.matmul(x, w_q, stream=s_q)
        k = mx.matmul(x, w_k, stream=s_q)
        v = mx.matmul(x, w_v, stream=s_q)
        mx.eval(q, k, v)
    t_three_one_stream = (time.perf_counter() - t0) / N_iter * 1e6

    print(f"shape  M={M}, K={K}, N={N}  bf16  ·  iters={N_iter}")
    print()
    print(f"A) 1 matmul · default stream:                  {t_single:7.1f} us  (baseline)")
    print(f"B) 3 matmul · default stream · 1 eval:         {t_three_default:7.1f} us  ({t_three_default/t_single:.2f}× baseline)")
    print(f"C) 3 matmul · default · mx.compile:            {t_compiled:7.1f} us  ({t_compiled/t_single:.2f}×)")
    print(f"G) 3 matmul · ONE new stream · 1 eval:         {t_three_one_stream:7.1f} us  ({t_three_one_stream/t_single:.2f}×)  ← sanity")
    print(f"E) 3 matmul · 3 STREAMS · 1 eval:              {t_three_streams:7.1f} us  ({t_three_streams/t_single:.2f}×)  ← target")
    print(f"F) 3 matmul · 3 STREAMS · each-eval:           {t_three_streams_eachsync:7.1f} us  ({t_three_streams_eachsync/t_single:.2f}×)")
    print()
    print(f"Verdict (E vs C):")
    if t_three_streams < t_compiled * 0.95:
        save_pct = (1 - t_three_streams / t_compiled) * 100
        print(f"  ★ 3-stream parallel WINS  by {save_pct:.1f} %  vs compiled — pursue integration")
    elif t_three_streams < t_compiled * 1.05:
        print(f"  3-stream ≈ compiled (within ±5%) — Apple already overlaps via default stream")
    else:
        regress = (t_three_streams / t_compiled - 1) * 100
        print(f"  ✗ 3-stream REGRESSES by {regress:.1f} % vs compiled — cross-stream sync overhead dominates")


if __name__ == "__main__":
    main()
