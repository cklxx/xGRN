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

## Custom Primitive subclass — wired, eval_gpu blocked on Metal Toolchain

`xgrn_ext.simdgroup_matmul_primitive(a, b)` is wired through
`xgrn_ext/cpp/simdgroup_matmul_primitive.{h,cpp}` as a `mlx::core::Primitive`
subclass named `SimdgroupMatmul`. mx.compile sees it as a NAMED primitive
in the graph, and shape inference works (`array.shape == (M, N)`,
`array.dtype == float32`). What is NOT implemented is `eval_gpu`.

### Why eval_gpu can't be implemented in this environment

Two approaches tried, both fail:

1. **Call `mx::fast::metal_kernel(...)(...)` from inside eval_gpu**, then
   `mx::eval` the result and `outputs[0].copy_shared_buffer(result)`.
   Crashes with a Metal assertion:
   ```
   -[_MTLCommandBuffer addCompletedHandler:]:1011:
   failed assertion `Completed handler provided after commit call'
   ```
   Forcing `mx::eval` inside an already-running primitive's eval_gpu
   creates a nested command-buffer lifecycle the Metal runtime rejects.

2. **Same trick with `mlx::core::matmul`** (i.e. just defer to MLX matmul
   inside eval_gpu) — same assertion. The problem isn't *which op*
   produces the result, it's the recursive `mx::eval` from eval_gpu.

The correct pattern, used by MLX's own primitives and the upstream
`examples/extensions/axpby`:

```cpp
void SimdgroupMatmul::eval_gpu(...) {
  outputs[0].set_data(allocator::malloc(nbytes));
  auto lib = metal::device(s).get_library("xgrn_ext", binary_dir());
  auto kernel = lib.get_kernel("xgrn_sgm_bf16");
  auto& encoder = metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  // ... bind buffers + dispatch ...
}
```

This requires a **precompiled `xgrn_ext.metallib`** built at extension
build time via the `mlx_build_metallib` CMake macro (from `extension.cmake`).
That macro shells out to `xcrun -sdk macosx metal`, and on this Mac:

```
$ xcrun -sdk macosx metal -c test.metal
error: cannot execute tool 'metal' due to missing Metal Toolchain;
use: xcodebuild -downloadComponent MetalToolchain
```

### To unblock

```bash
sudo xcodebuild -downloadComponent MetalToolchain
```

After that:
1. Add a `cpp/xgrn_ext.metal` with `xgrn_sgm_fp32` / `xgrn_sgm_bf16` kernels
2. Add `mlx_build_metallib(... TARGET xgrn_ext_metallib SOURCES cpp/xgrn_ext.metal ...)` to `CMakeLists.txt`
3. Implement `SimdgroupMatmul::eval_gpu` to load the metallib via
   `mlx::core::metal::device(s).get_library("xgrn_ext", binary_dir())`
   and dispatch via `metal::get_command_encoder`.

### Until then

Use `xgrn_ext.simdgroup_matmul_bf16` / `simdgroup_matmul_fp32` directly.
They dispatch via `mx::fast::metal_kernel` from C++ — opaque to
mx.compile but functionally correct and at parity with MLX matmul in
isolation.
