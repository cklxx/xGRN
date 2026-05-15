// Track A1-full: simdgroup_matmul implementation via mx::fast::metal_kernel.
//
// The kernel source mirrors xgrn_mlx/simdgroup_matmul.py exactly --
// 2x2 simdgroup layout + 2x4 register tile per simdgroup (8 fp32
// accumulators), output 32 rows x 64 cols per threadgroup. Loop
// reads 2 A tiles + 4 B tiles per K-step, does 8 multiply_accumulates.
// At M=544, K=2304, N=2304:
//   fp32: 1.045x of MLX matmul
//   bf16: 0.975x of MLX bf16 matmul  (2.5% faster than MLX)

#include "simdgroup_matmul.h"

#include <mlx/dtype.h>
#include <mlx/fast.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace xgrn_ext {

namespace {

const std::string kFp32Source = R"(
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
)";

const std::string kBf16Source = R"(
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
    simdgroup_matrix<float,  8, 8> Cm[8];
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
)";

const std::string kHeader = "#include <metal_simdgroup_matrix>\n";

// Cache the CustomKernelFunction so we don't reconstruct the kernel
// (and re-parse / re-key the source string) on every call. MLX caches
// by `name` internally but the outer std::function + vector<string>
// allocation here is real per-call overhead at 1400 qkv calls /
// generation.
mlx::core::fast::CustomKernelFunction& get_kernel(
    const std::string& name, const std::string& source) {
  static std::unordered_map<std::string, mlx::core::fast::CustomKernelFunction>
      cache;
  auto it = cache.find(name);
  if (it == cache.end()) {
    it = cache.emplace(
                  name,
                  mlx::core::fast::metal_kernel(
                      name,
                      /*input_names=*/{"A", "B"},
                      /*output_names=*/{"C"},
                      /*source=*/source,
                      /*header=*/kHeader,
                      /*ensure_row_contiguous=*/true,
                      /*atomic_outputs=*/false))
             .first;
  }
  return it->second;
}

mlx::core::array dispatch(
    const std::string& name,
    const std::string& source,
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::Dtype out_dtype,
    mlx::core::StreamOrDevice s) {
  using namespace mlx::core;
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul: requires 2D inputs");
  }
  int M = a.shape(0);
  int K = a.shape(1);
  int K2 = b.shape(0);
  int N = b.shape(1);
  if (K != K2) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul: inner dims must match");
  }
  if (M % 32 != 0 || N % 64 != 0 || K % 8 != 0) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul: M % 32 == 0, N % 64 == 0, K % 8 == 0 required");
  }
  auto& kernel = get_kernel(name, source);
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> templates = {
      {"M", M},
      {"K", K},
      {"N", N},
  };
  std::tuple<int, int, int> grid{128 * (M / 32) * (N / 64), 1, 1};
  std::tuple<int, int, int> threadgroup{128, 1, 1};
  auto outs = kernel(
      /*inputs=*/{a, b},
      /*output_shapes=*/{Shape{M, N}},
      /*output_dtypes=*/{out_dtype},
      /*grid=*/grid,
      /*threadgroup=*/threadgroup,
      /*template_args=*/templates,
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*s=*/s);
  return outs[0];
}

}  // namespace

mlx::core::array simdgroup_matmul_fp32(
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::StreamOrDevice s) {
  using namespace mlx::core;
  if (a.dtype() != float32 || b.dtype() != float32) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul_fp32: inputs must be fp32");
  }
  return dispatch("xgrn_ext_sgm_fp32", kFp32Source, a, b, float32, s);
}

mlx::core::array simdgroup_matmul_bf16(
    const mlx::core::array& a,
    const mlx::core::array& b,
    mlx::core::StreamOrDevice s) {
  using namespace mlx::core;
  if (a.dtype() != bfloat16 || b.dtype() != bfloat16) {
    throw std::invalid_argument(
        "xgrn_ext.simdgroup_matmul_bf16: inputs must be bf16");
  }
  return dispatch("xgrn_ext_sgm_bf16", kBf16Source, a, b, float32, s);
}

}  // namespace xgrn_ext
