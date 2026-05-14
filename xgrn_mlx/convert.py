from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import save_file


def _is_linear_weight(name: str, tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2


def _tensor_to_numpy(name: str, tensor: torch.Tensor, dtype: str, transpose_linear: bool) -> np.ndarray:
    if transpose_linear and _is_linear_weight(name, tensor):
        tensor = tensor.t().contiguous()
    if tensor.is_floating_point():
        if dtype == "fp16":
            tensor = tensor.to(torch.float16)
        elif dtype == "fp32":
            tensor = tensor.to(torch.float32)
        else:
            raise ValueError(f"unsupported dtype: {dtype}")
    return tensor.detach().cpu().numpy()


def _source_meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_torch_state(path: Path, root_key: str | None = None) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if root_key:
        state = state[root_key]
    elif isinstance(state, dict) and "ema" in state:
        state = state["ema"]
    elif isinstance(state, dict) and "vae" in state:
        state = state["vae"]
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not load to a state dict")
    return state


def convert_state(
    source: Path,
    output: Path,
    *,
    dtype: str = "fp16",
    transpose_linear: bool = True,
    root_key: str | None = None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    state = _load_torch_state(source, root_key=root_key)
    arrays: dict[str, np.ndarray] = {}
    converted = 0
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        arrays[name] = _tensor_to_numpy(name, tensor, dtype=dtype, transpose_linear=transpose_linear)
        converted += 1
    tmp = output.with_suffix(output.suffix + ".tmp")
    save_file(arrays, str(tmp))
    os.replace(tmp, output)
    return {
        "source": _source_meta(source),
        "output": str(output),
        "dtype": dtype,
        "transpose_linear": transpose_linear,
        "tensor_count": converted,
        "bytes": output.stat().st_size,
    }


def convert_all(model_dir: Path, out_dir: Path, dtype: str = "fp16") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "xgrn-mac-v1",
        "dtype": dtype,
        "artifacts": {},
    }
    for task, filename in [("t2i", "GRN_T2I_2B.pth"), ("t2v", "GRN_T2V_2B.pth")]:
        source = model_dir / filename
        if source.exists():
            manifest["artifacts"][f"grn_{task}"] = convert_state(
                source,
                out_dir / f"grn_{task}_{dtype}.safetensors",
                dtype=dtype,
                transpose_linear=True,
            )
    vae_source = model_dir / "HBQ_tokenizer_64dim_M4.ckpt"
    if vae_source.exists():
        manifest["artifacts"]["hbq"] = convert_state(
            vae_source,
            out_dir / f"hbq_{dtype}.safetensors",
            dtype=dtype,
            transpose_linear=False,
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official GRN weights to xGRN Mac runtime artifacts.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/GRN"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/GRN/mlx"))
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    manifest = convert_all(args.model_dir, args.out_dir, args.dtype)
    print(manifest)


if __name__ == "__main__":
    main()
