from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

from xgrn_mlx.bench import PROFILES, BenchProfile
from xgrn_mlx.validate import validate_image

from .t2i_mps import generate


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_profile(profile: BenchProfile, output_dir: Path, validate: bool, device: str, dtype: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_rss = _max_rss_mb()
    start = time.time()
    result = generate(
        task=profile.task,
        prompt=profile.prompt,
        negative_prompt=profile.negative_prompt,
        seed=profile.seed,
        pn=profile.pn,
        steps=profile.steps,
        guidance_scale=profile.guidance,
        temperature=profile.temperature,
        h_div_w=profile.h_div_w,
        duration=profile.duration,
        device_name=device,
        dtype_name=dtype,
    )
    end_to_end = time.time() - start
    image_src = Path("outputs/latest_t2i.png" if profile.task == "T2I" else "outputs/latest_t2v_first_frame.png")
    image_dst = output_dir / image_src.name
    if image_src.exists():
        image_dst.write_bytes(image_src.read_bytes())
    validation = validate_image(image_dst) if validate or profile.validate else None
    return {
        "runtime": "official-pytorch-mps",
        "device": device,
        "dtype": dtype,
        "profile": asdict(profile),
        "image": str(image_dst),
        "video": str(result.video) if result.video else None,
        "stats_last": result.stats[-1] if result.stats else None,
        "generation_reported_sec": result.elapsed_sec,
        "end_to_end_sec": end_to_end,
        "max_rss_mb": _max_rss_mb(),
        "rss_delta_mb": _max_rss_mb() - start_rss,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the original PyTorch/MPS GRN path.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="debug")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, mps, cpu, cuda.")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pytorch_bench"))
    parser.add_argument("--report", type=Path, default=Path("outputs/pytorch_bench/report.json"))
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    print(f"running original PyTorch/MPS profile={profile.name}", flush=True)
    row = run_profile(profile, args.output_dir / profile.name, args.validate, args.device, args.dtype)
    print(json.dumps(row, indent=2), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(row, indent=2))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
