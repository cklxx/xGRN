// Python bindings for xgrn_ext.
//
// Exposes simdgroup_matmul_{fp32,bf16} from C++ to Python via nanobind.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

#include "simdgroup_matmul.h"

namespace nb = nanobind;
using namespace mlx::core;

NB_MODULE(_core, m) {
  m.doc() = "xgrn_ext: simdgroup matmul as an MLX C++ extension";

  m.def(
      "simdgroup_matmul_fp32",
      [](const array& a, const array& b, StreamOrDevice s) {
        return xgrn_ext::simdgroup_matmul_fp32(a, b, s);
      },
      nb::arg("a"),
      nb::arg("b"),
      nb::kw_only(),
      nb::arg("stream") = nb::none(),
      R"pbdoc(
        Compute C = A @ B (fp32) via the 2x2 simdgroup + 2x4 register tile
        kernel exposed from C++. Same numerics + parity as
        xgrn_mlx.simdgroup_matmul.simdgroup_matmul_fp32, but dispatched
        from the C++ side so MLX's graph machinery handles the kernel
        directly.

        Constraints: A: [M, K] fp32, B: [K, N] fp32, M % 32 == 0,
        N % 64 == 0, K % 8 == 0.
      )pbdoc");

  m.def(
      "simdgroup_matmul_bf16",
      [](const array& a, const array& b, StreamOrDevice s) {
        return xgrn_ext::simdgroup_matmul_bf16(a, b, s);
      },
      nb::arg("a"),
      nb::arg("b"),
      nb::kw_only(),
      nb::arg("stream") = nb::none(),
      R"pbdoc(
        Compute C = A @ B with bf16 inputs and fp32 accumulator/output.
        Same kernel/layout as the fp32 variant; the inner simdgroup
        matrix is simdgroup_matrix<bfloat, 8, 8>, accumulator stays
        simdgroup_matrix<float, 8, 8>.

        Constraints: A: [M, K] bf16, B: [K, N] bf16, M % 32 == 0,
        N % 64 == 0, K % 8 == 0.
      )pbdoc");
}
