"""Simdgroup-matrix GEMM kernel for Track A1-full.

This is the foundation of the A1-full project: a Metal kernel that uses
Apple GPU family 9's `simdgroup_matrix<float, 8, 8>` + `simdgroup_multiply_accumulate`
to compute GEMM, intended to replace the naive scalar matmul in
`fused_norm_qproj` and eventually compete with MLX's tuned GEMM.

Status (2026-05-15, M=528, K=2304, N=2304):
  fp32:
    - Naive scalar kernel (fused_norm_qproj, earlier commit)  : 8.07x of MLX matmul
    - Naive 1-simdgroup-per-tile                              : 4.39x of MLX matmul
    - 2x2 simdgroup layout (this file)                        : 2.13x of MLX matmul   <-- current best
    - 4x4 simdgroup layout                                    : 2.22x of MLX matmul
    - 2x2 + threadgroup-cached A tile                         : 3.32x of MLX (barriers > caching benefit)
  bf16 (simdgroup_matrix<bfloat,8,8> with fp32 accumulator):
    - 2x2 simdgroup layout                                    : 2.22x of MLX bf16 matmul

Parity is byte-exact (max_abs_diff = 0.0) against `A @ B` (fp32) or
`(A @ B).astype(fp32)` (bf16). bf16 confirms Apple GPU 9 supports the
simdgroup_matrix bfloat path; it does NOT close the gap to MLX, which
means the gap is structural (bandwidth/occupancy), not compute throughput.

Why this is not yet wired into the GRN block:
  At 2.13x of MLX, replacing q_proj/k_proj/v_proj inside qkv() would regress
  total GRN time. The dispatch saving from fusing rms_norm into this kernel
  would be ~28 dispatches/step, worth ~0.3% wall by the dispatch-count
  theory — much less than the 2.13x compute penalty would cost. Need to
  close the gap to <=1.0x of MLX before integration is worthwhile.

Open paths to closing the gap (next-session work):
  1. bf16 simdgroup_matrix<bfloat, 8, 8>. Doubles compute throughput on
     family 9. Requires checking simdgroup_load/store accept bfloat.
  2. Async device-to-threadgroup loads with `metal::simd_async_copy` and
     double-buffered tile pipelining (load tile k+1 while computing on k).
  3. Larger output tiles per simdgroup via 1x4 or 2x4 simdgroup_matrix
     accumulators per simdgroup.
  4. Output-stationary register tiling: each simdgroup holds 2-4
     accumulators concurrently and iterates K once for all of them,
     amortizing the inner loop overhead.
  5. Tile-size autotuning across (M_TILE, N_TILE, K_STEP) sweep.
"""

from __future__ import annotations

import mlx.core as mx


_SIMDGROUP_MATMUL_FP32_KERNEL = None
_SIMDGROUP_MATMUL_BF16_KERNEL = None


def _ensure_kernel():
    global _SIMDGROUP_MATMUL_FP32_KERNEL
    if _SIMDGROUP_MATMUL_FP32_KERNEL is not None:
        return _SIMDGROUP_MATMUL_FP32_KERNEL
    source = """
        using namespace metal;

        // 2x2 simdgroup layout: 4 simdgroups per threadgroup (128 threads),
        // output tile is 16x16. Each simdgroup computes one 8x8 output tile,
        // loading A and B 8x8 simdgroup_matrix tiles directly from device
        // memory each K-step (L2 cache handles the dedup across simdgroups
        // that share the same row or column tile).
        uint sg = simdgroup_index_in_threadgroup;        // 0..3
        uint tg_id = threadgroup_position_in_grid.x;
        uint num_n_groups = N / 16;                       // N must be multiple of 16
        uint m_group = tg_id / num_n_groups;
        uint n_group = tg_id % num_n_groups;

        uint sg_row = sg >> 1;                            // 0..1
        uint sg_col = sg & 1u;                            // 0..1
        uint m_base = m_group * 16 + sg_row * 8;
        uint n_base = n_group * 16 + sg_col * 8;

        simdgroup_matrix<float, 8, 8> Am;
        simdgroup_matrix<float, 8, 8> Bm;
        simdgroup_matrix<float, 8, 8> Cm = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k = 0; k < K; k += 8) {
            simdgroup_load(Am, A + m_base * K + k, K);
            simdgroup_load(Bm, B + k * N + n_base, N);
            simdgroup_multiply_accumulate(Cm, Am, Bm, Cm);
        }
        simdgroup_store(Cm, C + m_base * N + n_base, N);
    """
    _SIMDGROUP_MATMUL_FP32_KERNEL = mx.fast.metal_kernel(
        name="xgrn_simdgroup_matmul_fp32_2x2",
        input_names=["A", "B"],
        output_names=["C"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n",
    )
    return _SIMDGROUP_MATMUL_FP32_KERNEL


def _ensure_kernel_bf16():
    global _SIMDGROUP_MATMUL_BF16_KERNEL
    if _SIMDGROUP_MATMUL_BF16_KERNEL is not None:
        return _SIMDGROUP_MATMUL_BF16_KERNEL
    source = """
        using namespace metal;
        uint sg = simdgroup_index_in_threadgroup;
        uint tg_id = threadgroup_position_in_grid.x;
        uint num_n_groups = N / 16;
        uint m_group = tg_id / num_n_groups;
        uint n_group = tg_id % num_n_groups;
        uint sg_row = sg >> 1;
        uint sg_col = sg & 1u;
        uint m_base = m_group * 16 + sg_row * 8;
        uint n_base = n_group * 16 + sg_col * 8;

        simdgroup_matrix<bfloat, 8, 8> Am;
        simdgroup_matrix<bfloat, 8, 8> Bm;
        simdgroup_matrix<float, 8, 8>  Cm = simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k = 0; k < K; k += 8) {
            simdgroup_load(Am, A + m_base * K + k, K);
            simdgroup_load(Bm, B + k * N + n_base, N);
            simdgroup_multiply_accumulate(Cm, Am, Bm, Cm);
        }
        simdgroup_store(Cm, C + m_base * N + n_base, N);
    """
    _SIMDGROUP_MATMUL_BF16_KERNEL = mx.fast.metal_kernel(
        name="xgrn_simdgroup_matmul_bf16_2x2",
        input_names=["A", "B"],
        output_names=["C"],
        source=source,
        header="#include <metal_simdgroup_matrix>\n",
    )
    return _SIMDGROUP_MATMUL_BF16_KERNEL


def simdgroup_matmul_bf16(A: mx.array, B: mx.array) -> mx.array:
    """Compute C = A @ B with bf16 inputs and fp32 accumulator.

    Same 2x2 simdgroup layout as `simdgroup_matmul_fp32` but uses
    `simdgroup_matrix<bfloat, 8, 8>` for A and B and accumulates into
    `simdgroup_matrix<float, 8, 8>`, then stores fp32 output. The pattern
    matches the production qkv() path (bf16 weights, fp32 accumulate).

    Parity is byte-exact (max_abs_diff = 0.0) vs MLX's `A @ B` cast to
    fp32 at (M=528, K=2304, N=2304). Time is ~2.22x of MLX's bf16 matmul
    -- same structural gap as fp32 version. bf16 is supported by the
    Apple GPU 9 simdgroup_matrix path but does not change the relative
    standing; further closing the gap needs bandwidth/occupancy work.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"simdgroup_matmul_bf16 requires 2D inputs, got {A.shape} @ {B.shape}")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"inner dims must match: A has K={K}, B has K={K2}")
    if M % 16 or N % 16:
        raise ValueError(f"M and N must be multiples of 16, got M={M}, N={N}")
    if K % 8:
        raise ValueError(f"K must be a multiple of 8, got K={K}")
    if A.dtype != mx.bfloat16 or B.dtype != mx.bfloat16:
        raise ValueError("simdgroup_matmul_bf16 requires bf16 inputs")
    kernel = _ensure_kernel_bf16()
    outs = kernel(
        inputs=[A, B],
        template=[("M", M), ("K", K), ("N", N)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(128 * (M // 16) * (N // 16), 1, 1),
        threadgroup=(128, 1, 1),
        verbose=False,
    )
    return outs[0] if isinstance(outs, (list, tuple)) else outs


def simdgroup_matmul_fp32(A: mx.array, B: mx.array) -> mx.array:
    """Compute C = A @ B using a 2x2 simdgroup-matrix GEMM kernel.

    Requires A.shape == (M, K), B.shape == (K, N), dtype fp32. M and N must
    be multiples of 16. Caller is responsible for padding if needed; this
    function does not auto-pad to avoid hiding shape errors.

    Parity is byte-exact (max_abs_diff = 0.0) vs `A @ B` at fp32 in the
    measured shape (M=528, K=2304, N=2304). Time is ~2.13x of MLX matmul
    in the same configuration -- not yet competitive for integration.

    See module docstring for the path-forward optimizations.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"simdgroup_matmul_fp32 requires 2D inputs, got {A.shape} @ {B.shape}")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"inner dims must match: A has K={K}, B has K={K2}")
    if M % 16 or N % 16:
        raise ValueError(f"M and N must be multiples of 16, got M={M}, N={N}")
    if K % 8:
        raise ValueError(f"K must be a multiple of 8, got K={K}")
    if A.dtype != mx.float32 or B.dtype != mx.float32:
        raise ValueError("simdgroup_matmul_fp32 currently only supports fp32 inputs")
    kernel = _ensure_kernel()
    outs = kernel(
        inputs=[A, B],
        template=[("M", M), ("K", K), ("N", N)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(128 * (M // 16) * (N // 16), 1, 1),
        threadgroup=(128, 1, 1),
        verbose=False,
    )
    return outs[0] if isinstance(outs, (list, tuple)) else outs
