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

## Custom Primitive subclass — landed end-to-end

`xgrn_ext.simdgroup_matmul_primitive(a, b)` is wired through
`xgrn_ext/cpp/simdgroup_matmul_primitive.{h,cpp}` as a
`mlx::core::Primitive` subclass named `SimdgroupMatmul`, with a working
axpby-style `eval_gpu` that dispatches against the precompiled
`xgrn_ext.metallib` (built at extension build time by MLX's
`mlx_build_metallib` macro, which requires the Metal Toolchain xcodebuild
component).

### What works

- `xgrn_ext.metallib` built from `cpp/xgrn_ext.metal` (12 KB)
- `eval_gpu` loads the metallib via `metal::device(s).get_library("xgrn_ext",
  binary_dir())`, allocates output via `out.set_data(allocator::malloc)`,
  binds inputs via `set_input_array` / `set_bytes`, dispatches via
  `dispatch_threads`
- Byte-exact parity vs the raw kernel and within bf16 floor vs MLX matmul
- mx.compile sees `SimdgroupMatmul` as a NAMED primitive in the graph

### What's NOT a free win

Integration into the GRN block via `--fuse-qkv-prim` measured a **regression**:
t2i-correct GRN +7.30 %, end-to-end +8.74 %, decode +45.66 %, CLIP
0.9904 → 0.9206. Worse than the raw-kernel path (`--fuse-qkv-ext` at
+3.77 % GRN, +15.23 % decode).

Why the axpby-style eval_gpu is slower per-call than `mx::fast::metal_kernel`:

- per-call `allocator::malloc` allocation (vs CustomKernel's pooled allocator)
- per-call `set_input_array` × 3 + `set_bytes` × 3 + `dispatch_threads`
- no encoder batching across calls

These add up to high overhead at 1400 qkv calls / generation. MLX's
internal CustomKernel infrastructure presumably amortizes these via
buffer-pool reuse and encoder batching that's not exposed in the public
extension API.

### Earlier blocker (now resolved)

For reference: `eval_gpu` was first attempted by calling
`mx::fast::metal_kernel(...)(...)` *from inside* eval_gpu, then
`mx::eval` on the result. That crashes the Metal runtime:

```
-[_MTLCommandBuffer addCompletedHandler:]:1011:
failed assertion `Completed handler provided after commit call'
```

Forcing `mx::eval` inside an already-running primitive's eval_gpu creates
a nested command-buffer lifecycle the Metal runtime rejects. The axpby
pattern avoids this by running everything on the parent stream's
existing encoder.

### Status

The Track A1-full infrastructure is now structurally complete across every
available framework level: raw `mx.fast.metal_kernel` (Python + C++),
manual MLX-only fusion, and axpby-style Custom Primitive subclass. None
deliver a net positive at the t2i-correct gate. The bottleneck is the
integration tax `mx.fast.metal_kernel`-class custom kernels pay inside
the visual_pass compile, regardless of how they're dispatched.
