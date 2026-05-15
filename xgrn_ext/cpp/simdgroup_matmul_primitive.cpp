// Custom Primitive subclass for SimdgroupMatmul.
//
// Status: SHAPE-INFERENCE ONLY. eval_gpu is blocked on the Metal Toolchain
// install (see the long comment block in eval_gpu below). The raw kernel
// in simdgroup_matmul.{h,cpp} works today and is what xgrn_mlx.grn calls
// via xgrn_ext.simdgroup_matmul_bf16. This subclass is wired in so that
// once the Metal Toolchain is installed and we can build an `.metallib`
// alongside, eval_gpu can be implemented properly (axpby-pattern).

#include "simdgroup_matmul_primitive.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/dtype.h>
#include <mlx/utils.h>

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace xgrn_ext {

namespace {

// Where is our .metallib? It sits next to the loaded _core.*.so.
std::string current_binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error(
          "xgrn_ext: dladdr failed -- cannot locate _core.so");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

}  // namespace

void SimdgroupMatmul::eval_cpu(
    const std::vector<mlx::core::array>& /*inputs*/,
    std::vector<mlx::core::array>& /*outputs*/) {
  throw std::runtime_error(
      "SimdgroupMatmul: CPU evaluation not implemented; use the fallback "
      "via mx.compile or evaluate on the Metal stream.");
}

void SimdgroupMatmul::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
  using namespace mlx::core;
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& out = outputs[0];

  uint32_t M = static_cast<uint32_t>(a.shape(0));
  uint32_t K = static_cast<uint32_t>(a.shape(1));
  uint32_t N = static_cast<uint32_t>(b.shape(1));

  // Allocate the output buffer in-place. axpby-pattern.
  out.set_data(allocator::malloc(out.nbytes()));

  // Load the precompiled metallib that sits next to _core.so.
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto lib = d.get_library("xgrn_ext", current_binary_dir());

  // Pick the right kernel by input dtype. Output is always fp32.
  const char* kname = (a.dtype() == bfloat16) ? "xgrn_sgm_bf16" : "xgrn_sgm_fp32";
  auto kernel = d.get_kernel(kname, lib);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(a, 0);
  enc.set_input_array(b, 1);
  enc.set_output_array(out, 2);
  enc.set_bytes(M, 3);
  enc.set_bytes(K, 4);
  enc.set_bytes(N, 5);

  // 2x2 simdgroup layout: 4 simdgroups per threadgroup x 32 lanes = 128 threads.
  // 32x64 output per threadgroup.
  const uint64_t threads_per_group = 128;
  const uint64_t num_groups =
      static_cast<uint64_t>(M / 32) * static_cast<uint64_t>(N / 64);
  MTL::Size group_dims = MTL::Size(threads_per_group, 1, 1);
  MTL::Size grid_dims = MTL::Size(threads_per_group * num_groups, 1, 1);
  enc.dispatch_threads(grid_dims, group_dims);
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
