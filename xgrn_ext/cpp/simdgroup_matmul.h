// Track A1-full: simdgroup_matmul as an MLX C++ extension.
//
// Same 2x2 simdgroup + 2x4 register-tile bf16 kernel as
// xgrn_mlx/simdgroup_matmul.py, but exposed via mx::fast::metal_kernel
// from inside the extension. Goal: get treated as a first-class
// primitive by the MLX runtime so the visual_pass mx.compile can
// schedule it alongside other ops without the integration tax we
// hit from Python-level mx.fast.metal_kernel calls.

#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace xgrn_ext {

// y = A @ B with fp32 inputs and fp32 output.
// Constraints: A: [M, K], B: [K, N], M % 32 == 0, N % 64 == 0, K % 8 == 0.
mlx::core::array simdgroup_matmul_fp32(
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::StreamOrDevice s = {});

// y = A @ B with bf16 inputs, fp32 accumulator + fp32 output.
mlx::core::array simdgroup_matmul_bf16(
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::StreamOrDevice s = {});

}  // namespace xgrn_ext
