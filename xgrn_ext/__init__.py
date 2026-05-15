"""xgrn_ext: MLX C++ extension shipping simdgroup matmul as a first-class op.

Usage:
    from xgrn_ext import simdgroup_matmul_bf16
    y = simdgroup_matmul_bf16(a_bf16, b_bf16)

Built via:
    cmake -S xgrn_ext -B xgrn_ext/build \\
        -DMLX_DIR=$(uv run python -c 'import mlx, os; print(os.path.dirname(mlx.__file__))')
    cmake --build xgrn_ext/build -j
"""

from ._core import (
    simdgroup_matmul_bf16,
    simdgroup_matmul_fp32,
    simdgroup_matmul_primitive,
)

__all__ = [
    "simdgroup_matmul_bf16",
    "simdgroup_matmul_fp32",
    "simdgroup_matmul_primitive",
]
