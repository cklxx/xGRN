// Custom Primitive subclass for SimdgroupMatmul.
//
// Status: SHAPE-INFERENCE ONLY. eval_gpu is blocked on the Metal Toolchain
// install (see the long comment block in eval_gpu below). The raw kernel
// in simdgroup_matmul.{h,cpp} works today and is what xgrn_mlx.grn calls
// via xgrn_ext.simdgroup_matmul_bf16. This subclass is wired in so that
// once the Metal Toolchain is installed and we can build an `.metallib`
// alongside, eval_gpu can be implemented properly (axpby-pattern).

#include "simdgroup_matmul_primitive.h"

#include <mlx/dtype.h>
#include <mlx/utils.h>

namespace xgrn_ext {

void SimdgroupMatmul::eval_cpu(
    const std::vector<mlx::core::array>& /*inputs*/,
    std::vector<mlx::core::array>& /*outputs*/) {
  throw std::runtime_error(
      "SimdgroupMatmul: CPU evaluation not implemented; use the fallback "
      "via mx.compile or evaluate on the Metal stream.");
}

void SimdgroupMatmul::eval_gpu(
    const std::vector<mlx::core::array>& /*inputs*/,
    std::vector<mlx::core::array>& /*outputs*/) {
  // *** BLOCKED on Metal Toolchain install. ***
  //
  // Two paths attempted, both fail on this machine:
  //   1. mx::fast::metal_kernel(...)(...) from inside eval_gpu, followed
  //      by mx::eval to materialize, then copy_shared_buffer. Crashes
  //      with a Metal assertion:
  //        "-[_MTLCommandBuffer addCompletedHandler:]:1011:
  //         Completed handler provided after commit call"
  //      because forcing mx::eval inside an already-running primitive's
  //      eval_gpu creates a nested command-buffer lifecycle the Metal
  //      runtime rejects.
  //   2. Same trick with mlx::core::matmul instead of metal_kernel.
  //      Same assertion -- the issue is the mx::eval call from inside
  //      eval_gpu, not which op produces the array.
  //
  // The correct pattern, demonstrated by MLX's own primitives and by
  // the upstream `axpby` example, is:
  //   - outputs[0].set_data(allocator::malloc(nbytes));
  //   - load the kernel from a precompiled .metallib via
  //     mlx::core::metal::device(s).get_library(name, path);
  //   - dispatch via metal::get_command_encoder(s);
  //
  // That requires the Metal Toolchain xcodebuild component which is
  // NOT installed on this Mac:
  //   $ xcrun -sdk macosx metal -c test.metal
  //   error: cannot execute tool 'metal' due to missing Metal Toolchain;
  //   use: xcodebuild -downloadComponent MetalToolchain
  //
  // Until that component is installed and `mlx_build_metallib` can
  // produce `xgrn_ext.metallib`, this primitive is a SHAPE-INFERENCE-ONLY
  // declaration. The Python facade still exposes the raw kernel via
  // simdgroup_matmul_bf16/fp32 which works fine -- those just don't get
  // the named-primitive treatment in mx.compile.
  throw std::runtime_error(
      "SimdgroupMatmul::eval_gpu: blocked on Metal Toolchain install. "
      "Use xgrn_ext.simdgroup_matmul_bf16 / simdgroup_matmul_fp32 instead; "
      "they dispatch via mx::fast::metal_kernel at the Python level and work "
      "today. See xgrn_ext/cpp/simdgroup_matmul_primitive.cpp eval_gpu "
      "comment block for the full diagnostic and the path forward.");
}

mlx::core::array simdgroup_matmul_primitive(
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::StreamOrDevice s_) {
  using namespace mlx::core;
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul_primitive: requires 2D inputs");
  }
  int M = a.shape(0);
  int K = a.shape(1);
  int K2 = b.shape(0);
  int N = b.shape(1);
  if (K != K2) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul_primitive: inner dims must match");
  }
  if (M % 32 != 0 || N % 64 != 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul_primitive: "
        "M % 32 == 0, N % 64 == 0, K % 8 == 0 required");
  }
  if (a.dtype() != b.dtype() ||
      (a.dtype() != float32 && a.dtype() != bfloat16)) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul_primitive: "
        "inputs must both be fp32 or both bf16");
  }

  auto s = to_stream(s_);
  return array(
      Shape{M, N},
      float32,
      std::make_shared<SimdgroupMatmul>(s, float32),
      {a, b});
}

}  // namespace xgrn_ext
