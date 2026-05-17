"""Benchmark silu(gate)*up — the swiglu element-wise op.

Three variants:
  A) eager:        silu(gate) * up  (two MLX ops)
  B) compiled:     mx.compile-wrapped silu(gate) * up
  C) fused Metal:  the existing xgrn_swiglu_fused kernel from xgrn_mlx.grn

If C beats B by a meaningful margin, flip fuse_swiglu_metal=True default.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from xgrn_mlx.grn import silu, swiglu


def main():
    # Per block, after gate/up matmuls:  [B*L, mlp_hidden] = [514, 8192]
    M, H = 514, 8192
    rng = np.random.RandomState(0)
    gate = mx.array(rng.randn(M, H).astype(np.float32) * 0.1).astype(mx.bfloat16)
    up   = mx.array(rng.randn(M, H).astype(np.float32) * 0.1).astype(mx.bfloat16)
    mx.eval(gate, up)

    N_iter = 500

    # --- A) eager: silu(gate) * up ---
    for _ in range(30):
        mx.eval(silu(gate) * up)
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(silu(gate) * up)
    t_eager = (time.perf_counter() - t0) / N_iter * 1e6

    # --- B) mx.compile wrapped ---
    @mx.compile
    def swiglu_compiled(g, u):
        return silu(g) * u

    for _ in range(30):
        mx.eval(swiglu_compiled(gate, up))
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(swiglu_compiled(gate, up))
    t_compiled = (time.perf_counter() - t0) / N_iter * 1e6

    # --- C) fused Metal kernel ---
    for _ in range(30):
        mx.eval(swiglu(gate, up))
    t0 = time.perf_counter()
    for _ in range(N_iter):
        mx.eval(swiglu(gate, up))
    t_fused = (time.perf_counter() - t0) / N_iter * 1e6

    # Sanity: max diff between fused vs eager (bf16 floor)
    diff = mx.max(mx.abs(swiglu(gate, up).astype(mx.float32) - (silu(gate) * up).astype(mx.float32)))
    mx.eval(diff)

    # IO calculation
    bytes_per_elem = 2  # bf16
    io_mb = (3 * M * H * bytes_per_elem) / 1e6  # read gate + read up + write out

    print(f"shape M={M}, H={H}  bf16  ·  iters={N_iter}")
    print(f"IO per call: {io_mb:.2f} MB  (= 2 reads + 1 write)")
    print()
    print(f"A) eager silu(gate)*up:               {t_eager:7.1f} us   {io_mb/t_eager*1e3:6.1f} GB/s")
    print(f"B) mx.compile silu(gate)*up:          {t_compiled:7.1f} us   {io_mb/t_compiled*1e3:6.1f} GB/s")
    print(f"C) fused Metal xgrn_swiglu_fused:     {t_fused:7.1f} us   {io_mb/t_fused*1e3:6.1f} GB/s")
    print()
    print(f"max |fused - eager| (fp32): {float(diff):.6f}")
    print()

    # Speedup vs production (which is currently B — silu*up inside compiled visual_pass)
    delta_vs_B = (t_fused / t_compiled - 1) * 100
    print(f"Verdict (C vs B production):  {delta_vs_B:+.1f}%")
    if delta_vs_B < -5:
        save_us = (t_compiled - t_fused)
        save_per_step_ms = save_us * 28 / 1000  # 28 blocks
        save_e2e_s = save_per_step_ms * 50 / 1000
        print(f"  ★ Fused wins.  Estimated step save: {save_per_step_ms:.2f} ms × 50 steps = {save_e2e_s:.2f} s e2e")
    else:
        print(f"  ✗ no meaningful win — mx.compile already fuses silu*up effectively")


if __name__ == "__main__":
    main()
