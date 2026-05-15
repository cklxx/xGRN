// xGRN Metal kernels for the C++ extension. Compiled to xgrn_ext.metallib
// by `mlx_build_metallib` (requires Metal Toolchain).
//
// Layout matches xgrn_mlx/simdgroup_matmul.py exactly: 2x2 simdgroup
// per threadgroup (128 threads), 2x4 register tile per simdgroup
// (8 fp32 accumulators), output 32 rows x 64 cols per threadgroup.
// Constraints: M % 32 == 0, N % 64 == 0, K % 8 == 0.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

[[kernel]] void xgrn_sgm_fp32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]) {

    uint num_n_groups = N / 64;
    uint m_group = tg_id.x / num_n_groups;
    uint n_group = tg_id.x % num_n_groups;
    uint sg_row = sg >> 1;
    uint sg_col = sg & 1u;
    uint m_base = m_group * 32 + sg_row * 16;
    uint n_base = n_group * 64 + sg_col * 32;

    simdgroup_matrix<float, 8, 8> Am[2], Bm[4];
    simdgroup_matrix<float, 8, 8> Cm[8];
    for (int i = 0; i < 8; ++i)
        Cm[i] = simdgroup_matrix<float, 8, 8>(0.0f);

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
}

[[kernel]] void xgrn_sgm_bf16(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]) {

    uint num_n_groups = N / 64;
    uint m_group = tg_id.x / num_n_groups;
    uint n_group = tg_id.x % num_n_groups;
    uint sg_row = sg >> 1;
    uint sg_col = sg & 1u;
    uint m_base = m_group * 32 + sg_row * 16;
    uint n_base = n_group * 64 + sg_col * 32;

    simdgroup_matrix<bfloat, 8, 8> Am[2], Bm[4];
    simdgroup_matrix<float,  8, 8> Cm[8];
    for (int i = 0; i < 8; ++i)
        Cm[i] = simdgroup_matrix<float, 8, 8>(0.0f);

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
}
