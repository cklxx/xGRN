from __future__ import annotations

import argparse
from pathlib import Path

from .t2i_mps import ROOT, download_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official GRN weights into the local xGRN model cache.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/GRN")
    parser.add_argument("--t2i-only", action="store_true", help="download T2I but skip the T2V transformer")
    args = parser.parse_args()
    path = download_weights(args.model_dir, include_t2v=not args.t2i_only)
    print(path)


if __name__ == "__main__":
    main()

