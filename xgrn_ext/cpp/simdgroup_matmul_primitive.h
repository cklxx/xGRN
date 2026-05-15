// SimdgroupMatmul as an mx::fast::Custom subclass.
//
// The earlier `simdgroup_matmul.{h,cpp}` exposed the kernel by calling
// mx::fast::metal_kernel(...) directly each invocation -- the resulting op
// is treated as opaque by mx.compile, which causes integration regressions
// (decode +88% etc.) when wired into the GRN block.
//
// This subclass approach gives the kernel:
//   - a NAMED primitive in the compile graph (DEFINE_NAME(SimdgroupMatmul))
//   - a fallback_ MLX op chain ((a @ b).astype(out_dtype)) that mx.compile
//     traces for shape inference / fusion
//   - eval_gpu that dispatches the actual kernel via mx::fast::metal_kernel,
//     copying the result via array::copy_shared_buffer
//
// Same 2x2 simdgroup + 2x4 register tile (8 accumulators) layout as the
// existing kernel. Constraints: M % 32 == 0, N % 64 == 0, K % 8 == 0,
// inputs fp32 or bf16.

#pragma once

#include <mlx/array.h>
#include <mlx/primitives.h>
#include <mlx/utils.h>

namespace xgrn_ext {

// We subclass Primitive directly (not mx::fast::Custom): Custom's
// vtable / typeinfo / vjp / vmap symbols are not exported by
// libmlx.dylib, so the linker rejects out-of-tree subclasses. The axpby
// example in MLX upstream uses the same pattern.
class SimdgroupMatmul : public mlx::core::Primitive {
 public:
  explicit SimdgroupMatmul(
      mlx::core::Stream stream, mlx::core::Dtype out_dtype)
      : mlx::core::Primitive(stream), out_dtype_(out_dtype) {}

  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  std::vector<mlx::core::array> jvp(
      const std::vector<mlx::core::array>& /*primals*/,
      const std::vector<mlx::core::array>& /*tangents*/,
      const std::vector<int>& /*argnums*/) override {
    throw std::runtime_error("SimdgroupMatmul has no jvp implementation.");
  }

  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& /*primals*/,
      const std::vector<mlx::core::array>& /*cotangents*/,
      const std::vector<int>& /*argnums*/,
      const std::vector<mlx::core::array>& /*outputs*/) override {
    throw std::runtime_error("SimdgroupMatmul has no vjp implementation.");
  }

  std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
      const std::vector<mlx::core::array>& /*inputs*/,
      const std::vector<int>& /*axes*/) override {
    throw std::runtime_error("SimdgroupMatmul has no vmap implementation.");
  }

  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    return {mlx::core::Shape{inputs[0].shape(0), inputs[1].shape(1)}};
  }

  const char* name() const override {
    return "SimdgroupMatmul";
  }

  bool is_equivalent(const mlx::core::Primitive& other) const override {
    auto& o = static_cast<const SimdgroupMatmul&>(other);
    return out_dtype_ == o.out_dtype_;
  }

 private:
  mlx::core::Dtype out_dtype_;
};

// y = matmul(a, b) cast to fp32. a and b must be 2D, M%32==0, N%64==0, K%8==0.
// dtype: bf16 or fp32.
mlx::core::array simdgroup_matmul_primitive(
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::StreamOrDevice s = {});

}  // namespace xgrn_ext
