# xgrn_ext — MLX C++ extension for xGRN

Custom MLX primitives built as a C++ extension, dispatching the
`simdgroup_matmul` kernel from inside the MLX runtime instead of via
the Python `mx.fast.metal_kernel` API. Track A1-full of the xGRN
performance plan.

## Status

The build pipeline works on a stock macOS+Xcode install (no Metal
Toolchain component required, because the kernel source is compiled
at runtime by MLX via `mx::fast::metal_kernel`, not at build time
with `xcrun metal`). Build artifact (`_core.cpython-311-darwin.so`) is
emitted to `xgrn_ext/_core.cpython-311-darwin.so` so the package
imports as `xgrn_ext` from project root.

Isolated microbench at M=544, K=2304, N=2304 (bf16, warm):
- MLX bf16 matmul:    ~1275 µs / call
- xgrn_ext bf16 (C++): ~1290 µs / call  (~1.01× of MLX, within noise)

For comparison, the same kernel called through Python's
`mx.fast.metal_kernel` runs at the same speed. The C++ path's
theoretical edge is in dispatch overhead amortization across many
calls, which our `t2i-correct` integration shows.

## Build

Requires:
- macOS + Xcode (clang compiler)
- cmake >= 3.27
- nanobind (installed via `uv add --dev nanobind`)
- mlx (already a runtime dependency)

```bash
cd <repo root>
MLX_PKG=$(uv run python -c "import mlx; print(mlx.__path__[0])")
PYTHON=$(uv run python -c "import sys; print(sys.executable)")
cmake -S xgrn_ext -B xgrn_ext/build \
  -DMLX_DIR="$MLX_PKG/share/cmake/MLX" \
  -DPython_EXECUTABLE="$PYTHON" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build xgrn_ext/build -j
```

The build is intentionally NOT triggered by `uv sync` because it
needs MLX installed first AND the build is the user's choice (the
runtime falls back to the Python `simdgroup_matmul.py` kernel if
the extension is missing).

## Layout

```
xgrn_ext/
  __init__.py            # Python facade: from ._core import ...
  cpp/
    simdgroup_matmul.h   # public C++ API
    simdgroup_matmul.cpp # implementation via mx::fast::metal_kernel
    bindings.cpp         # nanobind module (NB_DOMAIN mlx shares
                         # type-casters with MLX's own bindings)
  CMakeLists.txt
  _core.<abi>.so         # built artifact (gitignored)
  build/                 # cmake out-of-source build (gitignored)
```

## Usage

```python
import xgrn_ext
import mlx.core as mx

A = mx.zeros((544, 2304), dtype=mx.bfloat16)
B = mx.zeros((2304, 2304), dtype=mx.bfloat16)
C = xgrn_ext.simdgroup_matmul_bf16(A, B)  # [544, 2304] fp32
```

The extension is currently consumed by `xgrn_mlx.grn._qkv_fused_ext`
when the `--fuse-qkv-ext` runtime flag is on (default OFF).

## Next iteration: Custom Primitive subclass

The current C++ functions call `mx::fast::metal_kernel(...)` which
constructs an MLX `CustomKernel` primitive each call. Even with the
kernel cache the compile graph sees this as an opaque op and cannot
fuse around it -- the same limitation that hits the Python
`mx.fast.metal_kernel` path. A proper solution subclasses
`mx::fast::Custom` from `mlx/fast_primitives.h`:

```cpp
class SimdgroupMatmul : public mx::fast::Custom {
  // store M, K, N, dtype as members
  // override eval_gpu to call mx::fast::metal_kernel
  // provide a fallback `array @ array` lambda for mx.compile to trace
  // override output_shapes, name, is_equivalent, vjp/jvp stubs
};
```

The fallback lambda is what `mx.compile` uses for shape/dtype
inference and to compose with surrounding ops. With it in place the
kernel becomes a named primitive in the compile graph -- on par with
how `RMSNorm` and `RoPE` already work inside MLX.
