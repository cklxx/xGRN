"""Simdgroup-matrix GEMM kernel for Track A1-full.

Custom Metal kernel that uses Apple GPU family 9's `simdgroup_matrix<T, 8, 8>`
+ `simdgroup_multiply_accumulate` to compute GEMM. Combined with output-
stationary register tiling (2 M × 4 N = 8 accumulators per simdgroup), this
kernel matches or beats Apple's tuned MLX matmul on M4 Pro.

Status (2026-05-15, M=544 padded from 514, K=2304, N=2304):
    fp32:
      naive scalar kernel (`fused_norm_qproj` in grn.py)        : 8.07x of MLX
      naive 1-simdgroup-per-tile                                 : 4.39x
      2x2 simdgroup layout (1 accumulator each)                  : 2.13x
      4x4 simdgroup layout                                       : 2.22x
      2x2 + threadgroup-cached A tile                            : 3.32x (barriers > savings)
      2x2 simdgroup + 2 accumulators per simdgroup (horizontal)  : 1.82x
      2x2 simdgroup + 2x2 register tile per simdgroup (4 acc)    : 1.34x
      2x2 simdgroup + 4x4 register tile per simdgroup (16 acc)   : 6.32x (register spill)
      2x2 simdgroup + 4x2 register tile per simdgroup (8 acc)    : 1.09x
      2x2 simdgroup + 4x3 register tile per simdgroup (12 acc)   : 5.65x (register spill)
      2x2 simdgroup + 2x4 register tile per simdgroup (8 acc)    : 1.05x   <-- fp32 winner
    bf16 (simdgroup_matrix<bfloat,8,8> inputs, fp32 accumulator):
      MLX bf16 baseline                                          : 1.00x
      2x2 simdgroup + 2x4 register tile per simdgroup (8 acc)    : 0.95x   <-- bf16 winner

Parity: fp32 is byte-exact (max_abs_diff = 0.0). bf16 is within precision
floor (max_abs_diff ~3.86e-3, ~typical bf16 rounding accumulating over a
2304-dim dot product).

Layout summary (winner):
    Threadgroup = 4 simdgroups (128 threads) in 2x2 logical layout.
    Each simdgroup holds 8 accumulator tiles in a 2 (M) x 4 (N) register
    grid -> per-simdgroup output 16 rows x 32 cols.
    Per threadgroup output: 32 rows x 64 cols (2 sg-rows x 16 = 32 rows,
                                                  2 sg-cols x 32 = 64 cols).
    Per K-step (K stride 8): 2 A 8x8 simdgroup_loads, 4 B 8x8 simdgroup_loads,
    8 simdgroup_multiply_accumulates per simdgroup.
    Compute density: 8 mac / 6 loads = 1.33 mac/load (vs 1.0 for naive layout).

Why this beats MLX's bf16 matmul: register-tiled accumulators amortize
the inner-loop overhead AND keep the same A,B tile re-used across 8
accumulators in registers, where MLX's general-purpose kernel may use
fewer accumulators to support a wider range of shapes.

Open paths for further wins (next-iteration):
    1. metal::simdgroup_async_copy device->threadgroup pipelining --
       requires Metal 3 simdgroup_event header not present in current MLX
       toolchain; deferred until toolchain refresh.
    2. Larger M/N tiles via 3 sg-rows or 4 sg-cols (would need 3x N
       multiples or 4 row groups). Worth a small sweep.
    3. Specialize the inner loop for K=2304 specifically (constant-folded
       loop bound, software pipelining of the simdgroup_multiply).

When wired into the GRN block (fusing rms_norm), even matching MLX would
turn the dispatch savings into real wall reduction.
"""

from __future__ import annotations

import mlx.core as mx


_SIMDGROUP_MATMUL_FP32_KERNEL = None
_SIMDGROUP_MATMUL_BF16_KERNEL = None


_FP32_SOURCE = """
    using namespace metal;
    uint sg = simdgroup_index_in_threadgroup;
    uint tg_id = threadgroup_position_in_grid.x;
    uint num_n_groups = N / 64;
    uint m_group = tg_id / num_n_groups;
    uint n_group = tg_id % num_n_groups;
    uint sg_row = sg >> 1;
    uint sg_col = sg & 1u;
    uint m_base = m_group * 32 + sg_row * 16;
    uint n_base = n_group * 64 + sg_col * 32;

    simdgroup_matrix<float, 8, 8> Am[2], Bm[4];
    simdgroup_matrix<float, 8, 8> Cm[8];
    for (int i = 0; i < 8; ++i) Cm[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < K; k += 8) {
        for (int i = 0; i < 2; ++i)
            simdgroup_load(Am[i], A + (m_base + i * 8) * K + k, K);
        for (int j = 0; j < 4; ++j)
            simdgroup_load(Bm[j], B + k * N + n_base + j * 8, N);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(Cm[i * 4 + j], Am[i], Bm[j], Cm[i * 4 + j]);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            simdgroup_store(Cm[i * 4 + j], C + (m_base + i * 8) * N + n_base + j * 8, N);
"""


_BF16_SOURCE = """
    using namespace metal;
    uint sg = simdgroup_index_in_threadgroup;
    uint tg_id = threadgroup_position_in_grid.x;
    uint num_n_groups = N / 64;
    uint m_group = tg_id / num_n_groups;
    uint n_group = tg_id % num_n_groups;
    uint sg_row = sg >> 1;
    uint sg_col = sg & 1u;
    uint m_base = m_group * 32 + sg_row * 16;
    uint n_base = n_group * 64 + sg_col * 32;

    simdgroup_matrix<bfloat, 8, 8> Am[2], Bm[4];
    simdgroup_matrix<float, 8, 8>  Cm[8];
    for (int i = 0; i < 8; ++i) Cm[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < K; k += 8) {
        for (int i = 0; i < 2; ++i)
            simdgroup_load(Am[i], A + (m_base + i * 8) * K + k, K);
        for (int j = 0; j < 4; ++j)
            simdgroup_load(Bm[j], B + k * N + n_base + j * 8, N);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(Cm[i * 4 + j], Am[i], Bm[j], Cm[i * 4 + j]);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            simdgroup_store(Cm[i * 4 + j], C + (m_base + i * 8) * N + n_base + j * 8, N);
"""


def _ensure_kernel_fp32():
    global _SIMDGROUP_MATMUL_FP32_KERNEL
    if _SIMDGROUP_MATMUL_FP32_KERNEL is None:
        _SIMDGROUP_MATMUL_FP32_KERNEL = mx.fast.metal_kernel(
            name="xgrn_sgm_fp32_2x4reg",
            input_names=["A", "B"],
            output_names=["C"],
            source=_FP32_SOURCE,
            header="#include <metal_simdgroup_matrix>\n",
        )
    return _SIMDGROUP_MATMUL_FP32_KERNEL


def _ensure_kernel_bf16():
    global _SIMDGROUP_MATMUL_BF16_KERNEL
    if _SIMDGROUP_MATMUL_BF16_KERNEL is None:
        _SIMDGROUP_MATMUL_BF16_KERNEL = mx.fast.metal_kernel(
            name="xgrn_sgm_bf16_2x4reg",
            input_names=["A", "B"],
            output_names=["C"],
            source=_BF16_SOURCE,
            header="#include <metal_simdgroup_matrix>\n",
        )
    return _SIMDGROUP_MATMUL_BF16_KERNEL


def _check_shapes(A: mx.array, B: mx.array) -> tuple[int, int, int]:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"requires 2D inputs, got {A.shape} @ {B.shape}")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"inner dims must match: A has K={K}, B has K={K2}")
    if M % 32:
        raise ValueError(f"M must be a multiple of 32 (sg-row x register tile), got M={M}")
    if N % 64:
        raise ValueError(f"N must be a multiple of 64 (sg-col x register tile), got N={N}")
    if K % 8:
        raise ValueError(f"K must be a multiple of 8, got K={K}")
    return M, K, N


def simdgroup_matmul_fp32(A: mx.array, B: mx.array) -> mx.array:
    """Compute C = A @ B (fp32) using 2x2 simdgroup + 2x4 register tile.

    Layout: each simdgroup holds 8 accumulators (2 rows x 4 cols of 8x8
    tiles), threadgroup = 4 simdgroups in 2x2 -> per-group output 32x64.

    Microbench at M=544, K=2304, N=2304: ~1.05x of MLX matmul. Parity
    byte-exact (max_abs_diff = 0.0).

    Constraints: M multiple of 32, N multiple of 64, K multiple of 8.
    Caller pads if needed; this function does not auto-pad.
    """
    M, K, N = _check_shapes(A, B)
    if A.dtype != mx.float32 or B.dtype != mx.float32:
        raise ValueError("simdgroup_matmul_fp32 requires fp32 inputs")
    kernel = _ensure_kernel_fp32()
    outs = kernel(
        inputs=[A, B],
        template=[("M", M), ("K", K), ("N", N)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(128 * (M // 32) * (N // 64), 1, 1),
        threadgroup=(128, 1, 1),
        verbose=False,
    )
    return outs[0] if isinstance(outs, (list, tuple)) else outs


def simdgroup_matmul_bf16_padded(A: mx.array, B: mx.array) -> mx.array:
    """bf16 matmul with auto-padding when M is not a multiple of 32.

    The kernel requires M%32==0; this wrapper pads M up to the next multiple
    of 32 with zeros, calls `simdgroup_matmul_bf16`, slices the output back.
    Pad costs one dispatch; slice is a view. Useful when the matmul shape
    upstream is fixed by data (e.g. visual_tokens + scale_token = 257).
    """
    M, K = A.shape
    M_pad = (M + 31) // 32 * 32
    if M_pad == M:
        return simdgroup_matmul_bf16(A, B)
    A_padded = mx.pad(A, [(0, M_pad - M), (0, 0)])
    out = simdgroup_matmul_bf16(A_padded, B)
    return out[:M]


def simdgroup_matmul_bf16(A: mx.array, B: mx.array) -> mx.array:
    """Compute C = A @ B with bf16 inputs and fp32 accumulator.

    Same 2x2 simdgroup + 2x4 register tile layout as the fp32 variant.
    `simdgroup_matrix<bfloat, 8, 8>` for A and B, accumulating into
    `simdgroup_matrix<float, 8, 8>`, store fp32 output. Matches the
    production qkv() path (bf16 weights, fp32 accumulate).

    Microbench at M=544, K=2304, N=2304: ~0.95x of MLX bf16 matmul --
    that is, ~5 % faster than Apple's tuned MLX bf16 matmul. Parity is
    within bf16 precision (max_abs_diff ~3.86e-3 over a 2304-wide dot
    product).

    Constraints: M multiple of 32, N multiple of 64, K multiple of 8.
    """
    M, K, N = _check_shapes(A, B)
    if A.dtype != mx.bfloat16 or B.dtype != mx.bfloat16:
        raise ValueError("simdgroup_matmul_bf16 requires bf16 inputs")
    kernel = _ensure_kernel_bf16()
    outs = kernel(
        inputs=[A, B],
        template=[("M", M), ("K", K), ("N", N)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(128 * (M // 32) * (N // 64), 1, 1),
        threadgroup=(128, 1, 1),
        verbose=False,
    )
    return outs[0] if isinstance(outs, (list, tuple)) else outs
