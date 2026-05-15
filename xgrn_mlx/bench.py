from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .run import generate_mac
from .validate import validate_image


CAT_PROMPT = (
    "A realistic photo of an orange tabby cat sitting on a windowsill, fluffy fur, "
    "green eyes, soft daylight, natural indoor background, sharp focus, warm but realistic colors"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, distorted, deformed, extra limbs, "
    "cartoon, painting, illustration, oversaturated"
)


@dataclass(frozen=True)
class BenchProfile:
    name: str
    task: str
    prompt: str
    negative_prompt: str
    seed: int
    pn: str
    steps: int
    guidance: float
    temperature: float
    text_dtype: str = "bf16"
    text_cache_dtype: str = "fp32"
    weights_dtype: str = "fp32"
    compute_dtype: str = "bf16"
    decoder_backend: str = "native"
    decoder_weights_dtype: str = "fp16"
    decoder_compute_dtype: str = "fp16"
    h_div_w: float = 1.0
    duration: float = 0.25
    validate: bool = False


PROFILES = {
    "debug": BenchProfile(
        name="debug",
        task="T2I",
        prompt=CAT_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        pn="0.06M",
        steps=2,
        guidance=3.0,
        temperature=1.1,
    ),
    "t2i-correct": BenchProfile(
        name="t2i-correct",
        task="T2I",
        prompt=CAT_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        pn="0.25M",
        steps=50,
        guidance=3.0,
        temperature=1.1,
        validate=True,
    ),
    "t2i-step-stress": BenchProfile(
        name="t2i-step-stress",
        task="T2I",
        prompt=CAT_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        pn="0.06M",
        steps=150,
        guidance=3.0,
        temperature=1.1,
        validate=True,
    ),
    "t2i-quality-250": BenchProfile(
        name="t2i-quality-250",
        task="T2I",
        prompt=CAT_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        pn="0.25M",
        steps=250,
        guidance=3.0,
        temperature=1.1,
        validate=True,
    ),
    "t2v-short": BenchProfile(
        name="t2v-short",
        task="T2V",
        prompt="A realistic orange tabby cat turning its head in soft daylight, natural indoor background",
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        pn="0.06M",
        steps=16,
        guidance=3.0,
        temperature=1.1,
        duration=0.25,
    ),
}


def _max_rss_mb() -> float:
    # macOS returns ru_maxrss in bytes. Linux returns KiB; this project targets macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_profile(
    profile: BenchProfile,
    output_dir: Path,
    validate: bool,
    run_index: int = 0,
    compile_blocks: bool = False,
    compile_visual_pass: bool = True,
    compile_cfg_logits: bool = False,
    compile_refinement_update: bool = False,
    linear_quantization: str = "none",
    fuse_mlp_gate_up: bool = False,
    fuse_swiglu_metal: bool = False,
    fuse_rope_metal: bool = False,
    stack_cfg_cache: bool = False,
    detailed_stats: bool = False,
    exact_step_sync: bool = False,
    sampling_mode: str = "categorical",
    mask_schedule: str = "random",
    min_change_frac: float = 0.0,
    track_token_confidence: bool = False,
    precompute_pt_embed: bool = False,
    cfg_start_step: int = 0,
    release_after_run: bool = False,
    release_text_cache: bool = False,
) -> dict:
    start_rss = _max_rss_mb()
    start = time.perf_counter()
    result = generate_mac(
        task=profile.task,
        prompt=profile.prompt,
        negative_prompt=profile.negative_prompt,
        seed=profile.seed,
        pn=profile.pn,
        steps=profile.steps,
        guidance=profile.guidance,
        temperature=profile.temperature,
        text_dtype=profile.text_dtype,
        text_cache_dtype=profile.text_cache_dtype,
        weights_dtype=profile.weights_dtype,
        compute_dtype=profile.compute_dtype,
        decoder_backend=profile.decoder_backend,
        decoder_weights_dtype=profile.decoder_weights_dtype,
        decoder_compute_dtype=profile.decoder_compute_dtype,
        compile_blocks=compile_blocks,
        compile_visual_pass=compile_visual_pass,
        compile_cfg_logits=compile_cfg_logits,
        compile_refinement_update=compile_refinement_update,
        linear_quantization=linear_quantization,
        fuse_mlp_gate_up=fuse_mlp_gate_up,
        fuse_swiglu_metal=fuse_swiglu_metal,
        fuse_rope_metal=fuse_rope_metal,
        stack_cfg_cache=stack_cfg_cache,
        detailed_stats=detailed_stats,
        exact_step_sync=exact_step_sync,
        sampling_mode=sampling_mode,
        mask_schedule=mask_schedule,
        min_change_frac=min_change_frac,
        track_token_confidence=track_token_confidence,
        precompute_pt_embed=precompute_pt_embed,
        cfg_start_step=cfg_start_step,
        release_after_run=release_after_run,
        release_text_cache=release_text_cache,
        h_div_w=profile.h_div_w,
        duration=profile.duration,
        output_dir=output_dir,
    )
    generation_wall = time.perf_counter() - start
    image_path = output_dir / ("latest_t2i.png" if profile.task == "T2I" else "latest_t2v_first_frame.png")
    token_count = result.raw_shape[2] * result.raw_shape[3] * result.raw_shape[4] if len(result.raw_shape) == 5 else 0
    generated_tokens = token_count * profile.steps
    validation = None
    validation_sec = 0.0
    if validate or profile.validate:
        validation_start = time.perf_counter()
        validation = validate_image(image_path)
        validation_sec = time.perf_counter() - validation_start
    end_to_end = time.perf_counter() - start
    return {
        "profile": asdict(profile),
        "compile_blocks": compile_blocks,
        "compile_visual_pass": compile_visual_pass,
        "compile_cfg_logits": compile_cfg_logits,
        "compile_refinement_update": compile_refinement_update,
        "linear_quantization": linear_quantization,
        "fuse_mlp_gate_up": fuse_mlp_gate_up,
        "fuse_swiglu_metal": fuse_swiglu_metal,
        "fuse_rope_metal": fuse_rope_metal,
        "stack_cfg_cache": stack_cfg_cache,
        "detailed_stats": detailed_stats,
        "exact_step_sync": exact_step_sync,
        "sampling_mode": sampling_mode,
        "mask_schedule": mask_schedule,
        "min_change_frac": min_change_frac,
        "track_token_confidence": track_token_confidence,
        "precompute_pt_embed": precompute_pt_embed,
        "cfg_start_step": cfg_start_step,
        "release_after_run": release_after_run,
        "release_text_cache": release_text_cache,
        "run_index": run_index,
        "image": str(image_path),
        "video": str(result.video) if result.video else None,
        "raw_shape": result.raw_shape,
        "stats_last": result.stats[-1] if result.stats else None,
        "timings": result.timings,
        "generation_wall_sec": generation_wall,
        "validation_sec": validation_sec,
        "end_to_end_sec": end_to_end,
        "wall_sec": end_to_end,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": generated_tokens / result.timings["grn_refine_sec"] if result.timings.get("grn_refine_sec") else None,
        "max_rss_mb": _max_rss_mb(),
        "rss_delta_mb": _max_rss_mb() - start_rss,
        "validation": validation,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary: dict[str, dict[str, float | int]] = {}
    metric_paths = {
        "end_to_end_sec": ("end_to_end_sec",),
        "generation_wall_sec": ("generation_wall_sec",),
        "wall_sec": ("wall_sec",),
        "tokens_per_sec": ("tokens_per_sec",),
        "max_rss_mb": ("max_rss_mb",),
        "validation_sec": ("validation_sec",),
        "grn_refine_sec": ("timings", "grn_refine_sec"),
        "hbq_decode_sec": ("timings", "hbq_decode_sec"),
        "hbq_decode_compute_sec": ("timings", "hbq_decode_compute_sec"),
        "decoder_load_sec": ("timings", "decoder_load_sec"),
        "text_embeddings_sec": ("timings", "text_embeddings_sec"),
        "model_load_sec": ("timings", "model_load_sec"),
        "image_materialize_sec": ("timings", "image_materialize_sec"),
        "output_write_sec": ("timings", "output_write_sec"),
        "total_sec": ("timings", "total_sec"),
    }
    for name, path in metric_paths.items():
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        if values:
            summary[name] = {
                "count": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }
    validations = [row.get("validation") for row in rows if row.get("validation") is not None]
    if validations:
        summary["validation"] = {
            "count": len(validations),
            "passed": sum(1 for item in validations if item.get("passed")),
            "min_positive_score": min(float(item["positive_score"]) for item in validations),
            "median_positive_score": statistics.median(float(item["positive_score"]) for item in validations),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed xGRN Mac runtime profiles.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="debug")
    parser.add_argument("--all", action="store_true", help="Run all benchmark profiles.")
    parser.add_argument("--weights-dtype", choices=["fp32", "fp16", "auto"], help="Override profile GRN artifact dtype.")
    parser.add_argument("--text-dtype", choices=["bf16", "fp16", "fp32"], help="Override UMT5 text encoder dtype.")
    parser.add_argument("--text-cache-dtype", choices=["fp32", "fp16"], help="Override prompt embedding cache dtype.")
    parser.add_argument("--compute-dtype", choices=["fp32", "bf16", "fp16"], help="Override GRN matmul compute dtype.")
    parser.add_argument("--decoder-backend", choices=["native", "mps"], help="Override HBQ decoder backend.")
    parser.add_argument("--decoder-weights-dtype", choices=["fp32", "fp16"], help="Override native HBQ decoder artifact dtype.")
    parser.add_argument("--decoder-compute-dtype", choices=["fp32", "bf16", "fp16"], help="Override native HBQ decoder compute dtype.")
    parser.add_argument("--compile-blocks", action="store_true", help="Enable experimental mx.compile GRN block path.")
    parser.add_argument("--linear-quantization", choices=["none", "int8", "int4"], default="none", help="Experimental GRN linear weight-only quantized matmul.")
    parser.set_defaults(compile_visual_pass=True)
    parser.add_argument("--compile-visual-pass", dest="compile_visual_pass", action="store_true", help="Enable fixed-shape mx.compile over the full CFG visual pass.")
    parser.add_argument("--no-compile-visual-pass", dest="compile_visual_pass", action="store_false", help="Disable fixed-shape visual-pass compile.")
    parser.add_argument("--compile-cfg-logits", action="store_true", help="Compile visual pass plus CFG logits. Lower memory on the real gate, but slower than visual-pass compile.")
    parser.add_argument("--compile-refinement-update", action="store_true", help="Compile fixed-shape sampling and mask update after logits. Experimental.")
    parser.add_argument("--fuse-mlp-gate-up", action="store_true", help="Experimental: compute MLP gate/up projections as one wider matmul.")
    parser.add_argument("--fuse-swiglu-metal", action="store_true", help="Experimental: use a custom Metal kernel for silu(gate) * up.")
    parser.add_argument("--fuse-rope-metal", action="store_true", help="Experimental: fold the 7-dispatch apply_rope into one Metal kernel for Q/K rotation.")
    parser.add_argument("--stack-cfg-cache", action="store_true", help="Experimental: pass stacked CFG K/V cache tensors to the compiled visual pass.")
    parser.add_argument("--detailed-stats", action="store_true", help="Compute entropy and detailed per-step stats; slower because it syncs every step.")
    parser.add_argument("--exact-step-sync", action="store_true", help="Use sampled mask mean as the next step token, matching the debug parity path but adding a per-step sync.")
    parser.add_argument("--sampling-mode", choices=["categorical", "binary", "argmax"], default="categorical", help="Token sampling mode. categorical matches the official stochastic path; binary is an equivalent Bernoulli sampler for two classes; argmax is a debug quality tradeoff.")
    parser.add_argument("--mask-schedule", choices=["random", "dus"], default="random", help="Mask update schedule. random matches official behavior; dus is an experimental dilated unmasking schedule.")
    parser.add_argument("--min-change-frac", type=float, default=0.0, help="Experimental early-stop threshold. 0 disables early termination.")
    parser.add_argument("--track-token-confidence", action="store_true", help="Experimental: add per-step confidence stats for DUS research.")
    parser.add_argument("--precompute-pt-embed", action="store_true", help="Experimental: precompute per-step pt_embed tokens before refinement.")
    parser.add_argument(
        "--cfg-start-step",
        type=int,
        default=0,
        help=(
            "Skip the unconditional CFG lane for steps < K and resume CFG at step >= K. "
            "0 keeps current behavior (CFG every step, bit-identical)."
        ),
    )
    parser.add_argument("--steps", type=int, help="Override profile refinement steps.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each profile in one process to expose warm-run performance.")
    parser.add_argument("--warmup", action="store_true", help="Run each profile once before measurement in the same process.")
    parser.add_argument("--stable-shape-warmup", action="store_true", help="Warm the same shape with two refinement steps instead of running a full profile warmup.")
    parser.add_argument("--release-after-run", action="store_true", help="Release GRN/decoder/MLX caches after each measured run. Disables warm-cache benefit across repeats.")
    parser.add_argument("--release-text-cache", action="store_true", help="Also clear in-process prompt embedding arrays when --release-after-run is set.")
    parser.add_argument("--validate", action="store_true", help="Run CLIP validation even for profiles that do not require it.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bench"))
    parser.add_argument("--report", type=Path, default=Path("outputs/bench/report.json"))
    args = parser.parse_args()

    selected = list(PROFILES.values()) if args.all else [PROFILES[args.profile]]
    if args.text_dtype:
        selected = [replace(profile, text_dtype=args.text_dtype) for profile in selected]
    if args.text_cache_dtype:
        selected = [replace(profile, text_cache_dtype=args.text_cache_dtype) for profile in selected]
    if args.weights_dtype:
        selected = [replace(profile, weights_dtype=args.weights_dtype) for profile in selected]
    if args.compute_dtype:
        selected = [replace(profile, compute_dtype=args.compute_dtype) for profile in selected]
    if args.decoder_backend:
        selected = [replace(profile, decoder_backend=args.decoder_backend) for profile in selected]
    if args.decoder_weights_dtype:
        selected = [replace(profile, decoder_weights_dtype=args.decoder_weights_dtype) for profile in selected]
    if args.decoder_compute_dtype:
        selected = [replace(profile, decoder_compute_dtype=args.decoder_compute_dtype) for profile in selected]
    if args.steps is not None:
        selected = [replace(profile, steps=args.steps) for profile in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for profile in selected:
        if args.warmup or args.stable_shape_warmup:
            warmup_dir = args.output_dir / "_warmup" / profile.name
            warmup_dir.mkdir(parents=True, exist_ok=True)
            warmup_profile = replace(profile, steps=min(profile.steps, 2), validate=False) if args.stable_shape_warmup else profile
            print(
                f"warming profile={profile.name} steps={warmup_profile.steps} stable_shape={args.stable_shape_warmup}",
                flush=True,
            )
            run_profile(
                warmup_profile,
                warmup_dir,
                validate=False,
                run_index=-1,
                compile_blocks=args.compile_blocks,
                compile_visual_pass=args.compile_visual_pass,
                compile_cfg_logits=args.compile_cfg_logits,
                compile_refinement_update=args.compile_refinement_update,
                linear_quantization=args.linear_quantization,
                fuse_mlp_gate_up=args.fuse_mlp_gate_up,
                fuse_swiglu_metal=args.fuse_swiglu_metal,
                fuse_rope_metal=args.fuse_rope_metal,
                stack_cfg_cache=args.stack_cfg_cache,
                detailed_stats=args.detailed_stats,
                exact_step_sync=args.exact_step_sync,
                sampling_mode=args.sampling_mode,
                mask_schedule=args.mask_schedule,
                min_change_frac=args.min_change_frac,
                track_token_confidence=args.track_token_confidence,
                precompute_pt_embed=args.precompute_pt_embed,
                cfg_start_step=args.cfg_start_step,
                release_after_run=False,
                release_text_cache=False,
            )
        for run_index in range(args.repeat):
            profile_dir = args.output_dir / profile.name if args.repeat == 1 else args.output_dir / profile.name / f"run_{run_index}"
            profile_dir.mkdir(parents=True, exist_ok=True)
            print(f"running profile={profile.name} run={run_index}", flush=True)
            row = run_profile(
                profile,
                profile_dir,
                args.validate,
                run_index=run_index,
                compile_blocks=args.compile_blocks,
                compile_visual_pass=args.compile_visual_pass,
                compile_cfg_logits=args.compile_cfg_logits,
                compile_refinement_update=args.compile_refinement_update,
                linear_quantization=args.linear_quantization,
                fuse_mlp_gate_up=args.fuse_mlp_gate_up,
                fuse_swiglu_metal=args.fuse_swiglu_metal,
                fuse_rope_metal=args.fuse_rope_metal,
                stack_cfg_cache=args.stack_cfg_cache,
                detailed_stats=args.detailed_stats,
                exact_step_sync=args.exact_step_sync,
                sampling_mode=args.sampling_mode,
                mask_schedule=args.mask_schedule,
                min_change_frac=args.min_change_frac,
                track_token_confidence=args.track_token_confidence,
                precompute_pt_embed=args.precompute_pt_embed,
                cfg_start_step=args.cfg_start_step,
                release_after_run=args.release_after_run,
                release_text_cache=args.release_text_cache,
            )
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"runs": rows, "summary": summarize(rows)}
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
