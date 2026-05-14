from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xgrn_mlx.bootstrap import (
    BootstrapConfig,
    ModelBootstrapError,
    convert_dtypes_from_env,
    ensure_model_assets,
    model_dir_from_env,
    parse_convert_dtypes,
    repo_id_from_env,
    revision_from_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the local xGRN model cache. This uses HuggingFace snapshot_download, "
            "reuses existing files, retries a failed download once, and converts weights to MLX artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=model_dir_from_env(),
        help="Local GRN model cache directory. Env: XGRN_MODEL_DIR.",
    )
    parser.add_argument(
        "--repo-id",
        default=repo_id_from_env(),
        help="HuggingFace model repo id. Env: XGRN_HF_REPO_ID.",
    )
    parser.add_argument(
        "--revision",
        default=revision_from_env(),
        help="HuggingFace revision/tag/commit. Env: XGRN_HF_REVISION.",
    )
    parser.add_argument("--t2i-only", action="store_true", help="download T2I but skip the T2V transformer")
    parser.add_argument("--no-convert", action="store_true", help="Download raw HuggingFace files only; skip MLX conversion.")
    parser.add_argument(
        "--convert-dtypes",
        default=",".join(convert_dtypes_from_env()),
        help="Comma-separated MLX artifact dtypes to ensure when conversion is enabled. Env: XGRN_CONVERT_DTYPES.",
    )
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser()
    config = BootstrapConfig(
        model_dir=model_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        include_t2v=not args.t2i_only,
        auto_download=True,
        auto_convert=not args.no_convert,
        convert_dtypes=parse_convert_dtypes(args.convert_dtypes),
    )
    try:
        ensure_model_assets(config, progress=lambda msg: print(f"[xGRN] {msg}", flush=True))
    except ModelBootstrapError as exc:
        print(f"[xGRN] Download blocked:\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(model_dir)


if __name__ == "__main__":
    main()
