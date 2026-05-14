from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from .decode import HBQMPSDecoder
from .grn import GRN2BMLX
from .hbq_mlx import HBQMLXDecoder


REFERENCE_GRN = (Path(__file__).resolve().parents[1] / "../GRN").resolve()


def _ensure_reference_importable() -> None:
    root = str(REFERENCE_GRN)
    if root not in sys.path:
        sys.path.insert(0, root)


def compare_arrays(name: str, expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = expected.astype(np.float32)
    actual = actual.astype(np.float32)
    diff = np.abs(expected - actual)
    return {
        "name": name,
        "shape": list(expected.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "expected_mean": float(expected.mean()),
        "actual_mean": float(actual.mean()),
        "finite": bool(np.isfinite(actual).all()),
    }


def run_bit_label_parity(seed: int, pt: int, ph: int, pw: int) -> dict:
    _ensure_reference_importable()
    from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature

    torch.manual_seed(seed)
    labels_t = torch.randint(0, 2, (1, 256, pt, ph, pw), dtype=torch.long)
    raw_t = bit_label2raw_feature(labels_t, hbq_round=4)
    model = GRN2BMLX("models/GRN/mlx_fp32/grn_t2i_fp32.safetensors")
    raw_m = model.bit_labels_to_raw(mx.array(labels_t.numpy().astype(np.int32)))
    mx.eval(raw_m)
    return compare_arrays("bit_label2raw_feature", raw_t.numpy(), np.array(raw_m))


def run_decoder_parity(
    *,
    seed: int,
    pt: int,
    ph: int,
    pw: int,
    model_dir: Path,
    weights_dtype: str,
    compute_dtype: str,
) -> dict:
    mx.random.seed(seed)
    raw = mx.random.normal((1, 64, pt, ph, pw)).astype(mx.float16)
    mx.eval(raw)

    mps = HBQMPSDecoder(model_dir)
    start = time.perf_counter()
    expected = mps.decode_tensor(raw)
    mps_sec = time.perf_counter() - start

    native = HBQMLXDecoder(model_dir, weights_dtype=weights_dtype, compute_dtype=compute_dtype)
    start = time.perf_counter()
    actual = native.decode_tensor(raw)
    mx.eval(actual)
    native_sec = time.perf_counter() - start

    row = compare_arrays("hbq_decode", expected.numpy(), np.array(actual))
    row["mps_sec"] = mps_sec
    row["native_sec"] = native_sec
    row["speedup"] = mps_sec / native_sec if native_sec else None
    row["weights_dtype"] = weights_dtype
    row["compute_dtype"] = compute_dtype
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Check native MLX HBQ decoder against official PyTorch/MPS.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pt", type=int, default=1)
    parser.add_argument("--ph", type=int, default=16)
    parser.add_argument("--pw", type=int, default=16)
    parser.add_argument("--model-dir", type=Path, default=Path("models/GRN"))
    parser.add_argument("--weights-dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--compute-dtype", choices=["fp32", "bf16", "fp16"], default="fp16")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = {
        "bit_label": run_bit_label_parity(args.seed, args.pt, args.ph, args.pw),
        "decoder": run_decoder_parity(
            seed=args.seed,
            pt=args.pt,
            ph=args.ph,
            pw=args.pw,
            model_dir=args.model_dir,
            weights_dtype=args.weights_dtype,
            compute_dtype=args.compute_dtype,
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)


if __name__ == "__main__":
    main()
