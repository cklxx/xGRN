from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import mlx.core as mx
import numpy as np
from PIL import Image

from .bootstrap import (
    BootstrapConfig,
    ModelBootstrapError,
    convert_dtypes_from_env,
    dtypes_for_runtime,
    env_bool,
    ensure_runtime_ready,
    model_dir_from_env,
    parse_convert_dtypes,
    repo_id_from_env,
    revision_from_env,
)
from .decode import HBQMPSDecoder
from .grn import GRN2BMLX
from .hbq_mlx import HBQMLXDecoder
from .schedule import scale_schedule
from .text import clear_text_embedding_cache, load_or_create_text_embeddings_with_key


@dataclass
class MacGenerationResult:
    image: Image.Image
    video: Path | None
    stats: list[dict]
    raw_shape: tuple[int, ...]
    elapsed_sec: float
    timings: dict[str, float]
    refinement_frames: list[Image.Image] = field(default_factory=list)
    refinement_gif: Path | None = None


_MODEL_CACHE: dict[str, GRN2BMLX] = {}
_DECODER_CACHE: HBQMPSDecoder | None = None
_NATIVE_DECODER_CACHE: dict[str, HBQMLXDecoder] = {}


def _model(
    task: str,
    model_dir: Path,
    weights_dtype: str = "fp32",
    compute_dtype: str = "bf16",
    compile_blocks: bool = False,
    compile_visual_pass: bool = True,
    compile_cfg_logits: bool = False,
    compile_refinement_update: bool = False,
    linear_quantization: str = "none",
    fuse_mlp_gate_up: bool = False,
    fuse_swiglu_metal: bool = False,
    fuse_rope_metal: bool = False,
    fuse_residual_norm_metal: bool = False,
    fuse_qkv_metal: bool = False,
    fuse_qkv_concat: bool = False,
    stack_cfg_cache: bool = False,
) -> GRN2BMLX:
    task = task.lower()
    key = (
        f"{model_dir}:{task}:{weights_dtype}:{compute_dtype}:"
        f"linear_quant={linear_quantization}:compile={compile_blocks}:"
        f"compile_visual={compile_visual_pass}:compile_cfg_logits={compile_cfg_logits}:"
        f"compile_update={compile_refinement_update}:fuse_mlp_gate_up={fuse_mlp_gate_up}:"
        f"fuse_swiglu_metal={fuse_swiglu_metal}:fuse_rope_metal={fuse_rope_metal}:"
        f"fuse_residual_norm_metal={fuse_residual_norm_metal}:"
        f"fuse_qkv_metal={fuse_qkv_metal}:fuse_qkv_concat={fuse_qkv_concat}:"
        f"stack_cfg_cache={stack_cfg_cache}"
    )
    if key not in _MODEL_CACHE:
        fp32_path = model_dir / "mlx_fp32" / f"grn_{task}_fp32.safetensors"
        fp16_path = model_dir / "mlx" / f"grn_{task}_fp16.safetensors"
        if weights_dtype == "auto":
            weights_path = fp32_path if fp32_path.exists() else fp16_path
        elif weights_dtype == "fp32":
            weights_path = fp32_path
        elif weights_dtype == "fp16":
            weights_path = fp16_path
        else:
            raise ValueError(f"unsupported weights dtype {weights_dtype}")
        if not weights_path.exists():
            raise FileNotFoundError(f"missing GRN weights artifact: {weights_path}")
        _MODEL_CACHE[key] = GRN2BMLX(
            weights_path,
            compute_dtype=compute_dtype,
            compile_blocks=compile_blocks,
            compile_visual_pass=compile_visual_pass,
            compile_cfg_logits=compile_cfg_logits,
            compile_refinement_update=compile_refinement_update,
            linear_quantization=linear_quantization,
            fuse_mlp_gate_up=fuse_mlp_gate_up,
            fuse_swiglu_metal=fuse_swiglu_metal,
            fuse_rope_metal=fuse_rope_metal,
            fuse_residual_norm_metal=fuse_residual_norm_metal,
            fuse_qkv_metal=fuse_qkv_metal,
            fuse_qkv_concat=fuse_qkv_concat,
            stack_cfg_cache=stack_cfg_cache,
        )
    return _MODEL_CACHE[key]


def _decoder(model_dir: Path) -> HBQMPSDecoder:
    global _DECODER_CACHE
    if _DECODER_CACHE is None:
        _DECODER_CACHE = HBQMPSDecoder(model_dir)
    return _DECODER_CACHE


def _native_decoder(model_dir: Path, weights_dtype: str = "fp16", compute_dtype: str = "fp16") -> HBQMLXDecoder:
    key = f"{model_dir}:{weights_dtype}:{compute_dtype}"
    if key not in _NATIVE_DECODER_CACHE:
        _NATIVE_DECODER_CACHE[key] = HBQMLXDecoder(model_dir, weights_dtype=weights_dtype, compute_dtype=compute_dtype)
    return _NATIVE_DECODER_CACHE[key]


def clear_runtime_caches(clear_text: bool = False) -> None:
    global _DECODER_CACHE
    for model in _MODEL_CACHE.values():
        model.close()
    _MODEL_CACHE.clear()
    if _DECODER_CACHE is not None:
        _DECODER_CACHE.close()
        _DECODER_CACHE = None
    for decoder in _NATIVE_DECODER_CACHE.values():
        decoder.close()
    _NATIVE_DECODER_CACHE.clear()
    if clear_text:
        clear_text_embedding_cache()
    gc.collect()
    mx.clear_cache()


def _save_stats(stats: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not stats:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)


def _tensor_to_video(tensor, output: Path, fps: int = 16) -> Path:
    # tensor: [B, C, T, H, W], RGB in [0, 1]
    if isinstance(tensor, mx.array):
        arr = np.array(mx.transpose(tensor[0], (1, 2, 3, 0)))
    else:
        arr = tensor[0].permute(1, 2, 3, 0).numpy()
    arr = np.round(arr * 255).clip(0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, arr, fps=fps)
    return output


def _tensor_first_frame_to_image(tensor) -> Image.Image:
    if isinstance(tensor, mx.array):
        frame = np.array(mx.transpose(tensor[0, :, 0], (1, 2, 0)))
    else:
        frame = tensor[0, :, 0].permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(frame * 255).clip(0, 255).astype(np.uint8), mode="RGB")


def generate_mac(
    *,
    task: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    pn: str = "0.25M",
    steps: int = 50,
    guidance: float = 3.0,
    temperature: float = 1.1,
    h_div_w: float = 1.0,
    duration: float = 0.25,
    model_dir: Path = Path("models/GRN"),
    text_dtype: str = "bf16",
    text_cache_dtype: str = "fp32",
    weights_dtype: str = "fp32",
    compute_dtype: str = "bf16",
    decoder_backend: str = "native",
    decoder_weights_dtype: str = "fp16",
    decoder_compute_dtype: str = "fp16",
    compile_blocks: bool = False,
    compile_visual_pass: bool = True,
    compile_cfg_logits: bool = False,
    compile_refinement_update: bool = False,
    linear_quantization: str = "none",
    fuse_mlp_gate_up: bool = False,
    fuse_swiglu_metal: bool = False,
    fuse_rope_metal: bool = False,
    fuse_residual_norm_metal: bool = False,
    fuse_qkv_metal: bool = False,
    fuse_qkv_concat: bool = False,
    stack_cfg_cache: bool = False,
    detailed_stats: bool = False,
    exact_step_sync: bool = False,
    sampling_mode: str = "categorical",
    mask_schedule: str = "random",
    min_change_frac: float = 0.0,
    track_token_confidence: bool = False,
    precompute_pt_embed: bool = False,
    cfg_start_step: int = 0,
    capture_interval: int = 0,
    release_after_run: bool = False,
    release_text_cache: bool = False,
    output_dir: Path = Path("outputs"),
    progress: Callable[[str], None] | None = None,
) -> MacGenerationResult:
    start = time.perf_counter()
    timings: dict[str, float] = {}
    task = task.upper()
    if task not in {"T2I", "T2V"}:
        raise ValueError(f"unsupported task {task}")
    if decoder_backend not in {"native", "mps"}:
        raise ValueError(f"unsupported decoder backend {decoder_backend}")
    if progress:
        progress("loading prompt embeddings")
    stage = time.perf_counter()
    cond, uncond, prompt_cache_key = load_or_create_text_embeddings_with_key(
        prompt,
        negative_prompt,
        model_dir=model_dir,
        text_dtype=text_dtype,
        cache_dtype=text_cache_dtype,
    )
    timings["text_embeddings_sec"] = time.perf_counter() - stage
    num_frames = 1 if task == "T2I" else min(81, max(5, int(duration * 16 + 1)))
    schedule, mapped = scale_schedule(pn, h_div_w, num_frames)
    pt, ph, pw = schedule[0]
    stage = time.perf_counter()
    model = _model(
        task.lower(),
        model_dir,
        weights_dtype=weights_dtype,
        compute_dtype=compute_dtype,
        compile_blocks=compile_blocks,
        compile_visual_pass=compile_visual_pass,
        compile_cfg_logits=compile_cfg_logits,
        compile_refinement_update=compile_refinement_update,
        linear_quantization=linear_quantization,
        fuse_mlp_gate_up=fuse_mlp_gate_up,
        fuse_swiglu_metal=fuse_swiglu_metal,
        fuse_rope_metal=fuse_rope_metal,
        fuse_residual_norm_metal=fuse_residual_norm_metal,
        fuse_qkv_metal=fuse_qkv_metal,
        fuse_qkv_concat=fuse_qkv_concat,
        stack_cfg_cache=stack_cfg_cache,
    )
    timings["model_load_sec"] = time.perf_counter() - stage
    if progress:
        progress(
            "running MLX GRN: "
            f"task={task} weights={weights_dtype} compute={compute_dtype} "
            f"linear_quant={linear_quantization} "
            f"compile={compile_blocks} compile_visual={compile_visual_pass} "
            f"compile_cfg_logits={compile_cfg_logits} compile_update={compile_refinement_update} "
            f"fuse_mlp_gate_up={fuse_mlp_gate_up} "
            f"fuse_swiglu_metal={fuse_swiglu_metal} "
            f"stack_cfg_cache={stack_cfg_cache} "
            f"sampling={sampling_mode} scale=({pt},{ph},{pw}) steps={steps}"
        )

    # Init decoder early so frame_callback can use it during refinement.
    stage = time.perf_counter()
    decoder = (
        _native_decoder(model_dir, weights_dtype=decoder_weights_dtype, compute_dtype=decoder_compute_dtype)
        if decoder_backend == "native"
        else _decoder(model_dir)
    )
    timings["decoder_load_sec"] = time.perf_counter() - stage

    captured_frames: list[Image.Image] = []

    def on_frame(step: int, raw_frame: mx.array) -> None:
        frame_tensor = decoder.decode_tensor(raw_frame)
        captured_frames.append(_tensor_first_frame_to_image(frame_tensor))

    def step_progress(s: dict) -> None:
        if not progress:
            return
        entropy = s.get("entropy")
        if entropy is None:
            progress(f"step {s['step']}/{steps}: signal={s['next_pt']:.3f}")
        else:
            progress(f"step {s['step']}/{steps}: entropy={entropy:.3f} signal={s['next_pt']:.3f}")

    stage = time.perf_counter()
    raw, stats = model.refine(
        cond,
        uncond if guidance != 1.0 else None,
        pt=pt,
        ph=ph,
        pw=pw,
        mapped_h_div_w=mapped,
        steps=int(steps),
        guidance=float(guidance),
        temperature=float(temperature),
        seed=int(seed),
        progress=step_progress if progress else None,
        cond_cache_key=f"{prompt_cache_key}:cond",
        uncond_cache_key=f"{prompt_cache_key}:uncond",
        detailed_stats=detailed_stats,
        exact_step_sync=exact_step_sync,
        sampling_mode=sampling_mode,
        mask_schedule=mask_schedule,
        min_change_frac=min_change_frac,
        track_token_confidence=track_token_confidence,
        precompute_pt_embed=precompute_pt_embed,
        cfg_start_step=cfg_start_step,
        frame_callback=on_frame if capture_interval > 0 else None,
        capture_interval=capture_interval,
    )
    mx.eval(raw)
    timings["grn_refine_sec"] = time.perf_counter() - stage
    if progress:
        progress(f"decoding HBQ output: backend={decoder_backend}")
    decode_stage = time.perf_counter()
    stage = time.perf_counter()
    tensor = decoder.decode_tensor(raw)
    timings["hbq_decode_compute_sec"] = time.perf_counter() - stage
    timings["hbq_decode_sec"] = time.perf_counter() - decode_stage
    stage = time.perf_counter()
    image = _tensor_first_frame_to_image(tensor)
    timings["image_materialize_sec"] = time.perf_counter() - stage
    stage = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / ("latest_t2i.png" if task == "T2I" else "latest_t2v_first_frame.png")
    image.save(image_path)
    video_path = None
    if task == "T2V":
        video_path = _tensor_to_video(tensor, output_dir / "latest_t2v.mp4")
    _save_stats(stats, output_dir / "refinement_stats.csv")
    refinement_gif: Path | None = None
    if captured_frames:
        refinement_gif = output_dir / "latest_refinement.gif"
        captured_frames[0].save(
            refinement_gif,
            save_all=True,
            append_images=captured_frames[1:],
            loop=0,
            duration=200,
        )
    timings["output_write_sec"] = time.perf_counter() - stage
    timings["end_to_end_sec"] = time.perf_counter() - start
    timings["total_sec"] = timings["end_to_end_sec"]
    if release_after_run:
        clear_runtime_caches(clear_text=release_text_cache)
    return MacGenerationResult(
        image=image,
        video=video_path,
        stats=stats,
        raw_shape=tuple(raw.shape),
        elapsed_sec=timings["end_to_end_sec"],
        timings=timings,
        refinement_frames=captured_frames,
        refinement_gif=refinement_gif,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one xGRN generation on the Mac-specialized MLX runtime. "
            "Missing model assets are downloaded and converted before generation unless disabled."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["T2I", "T2V"], default="T2I")
    parser.add_argument("--prompt", default="A small red apple on a wooden table, natural light")
    parser.add_argument("--negative-prompt", default="blurry, low quality, watermark, text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pn", default="0.25M")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--text-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--text-cache-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--weights-dtype", choices=["fp32", "fp16", "auto"], default="fp32")
    parser.add_argument("--compute-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--decoder-backend", choices=["native", "mps"], default="native")
    parser.add_argument("--decoder-weights-dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--decoder-compute-dtype", choices=["fp32", "bf16", "fp16"], default="fp16")
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
    parser.add_argument(
        "--convert-dtypes",
        default=",".join(convert_dtypes_from_env()),
        help="Comma-separated MLX artifact dtypes to ensure. Env: XGRN_CONVERT_DTYPES.",
    )
    parser.add_argument(
        "--auto-download",
        dest="auto_download",
        action=argparse.BooleanOptionalAction,
        default=env_bool("XGRN_AUTO_DOWNLOAD", True),
        help="Download missing raw weights; --no-auto-download prints a repair command instead.",
    )
    parser.add_argument(
        "--auto-convert",
        dest="auto_convert",
        action=argparse.BooleanOptionalAction,
        default=env_bool("XGRN_AUTO_CONVERT", True),
        help="Create missing MLX artifacts; --no-auto-convert prints conversion commands instead.",
    )
    parser.add_argument("--skip-bootstrap", action="store_true", help="Run without checking model assets first.")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--linear-quantization", choices=["none", "int8", "int4"], default="none", help="Experimental GRN linear weight-only quantized matmul.")
    parser.set_defaults(compile_visual_pass=True)
    parser.add_argument("--compile-visual-pass", dest="compile_visual_pass", action="store_true", help="Enable fixed-shape mx.compile over the full CFG visual pass.")
    parser.add_argument("--no-compile-visual-pass", dest="compile_visual_pass", action="store_false", help="Disable fixed-shape visual-pass compile.")
    parser.add_argument("--compile-cfg-logits", action="store_true", help="Compile visual pass plus CFG logits. Lower memory on the real gate, but slower than visual-pass compile.")
    parser.add_argument("--compile-refinement-update", action="store_true", help="Compile fixed-shape sampling and mask update after logits. Experimental.")
    parser.add_argument("--fuse-mlp-gate-up", action="store_true", help="Experimental: compute MLP gate/up projections as one wider matmul.")
    parser.add_argument("--fuse-swiglu-metal", action="store_true", help="Experimental: use a custom Metal kernel for silu(gate) * up.")
    parser.add_argument("--fuse-rope-metal", action="store_true", help="Experimental: fold the 7-dispatch apply_rope into one Metal kernel for Q/K rotation.")
    parser.add_argument("--fuse-residual-norm-metal", action="store_true", help="Experimental: fold (x + attn) and post-attn rms_norm into one Metal kernel per block.")
    parser.add_argument("--fuse-qkv-metal", action="store_true", help="Track A1-full M3/M5: route q/k/v projection matmuls through simdgroup_matmul_bf16; saves the bf16->fp32 post-cast per matmul. Requires --compute-dtype bf16 and no linear quantization.")
    parser.add_argument("--fuse-qkv-concat", action="store_true", help="Track A1-full M3 alt: one MLX matmul against the concatenated [q|k|v] weight per block (no custom kernel; saves 2 matmul dispatches per qkv).")
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
    parser.add_argument("--release-after-run", action="store_true", help="Release GRN/decoder/MLX caches after writing outputs. Slower for repeated prompts but lowers post-run memory.")
    parser.add_argument("--release-text-cache", action="store_true", help="Also clear in-process prompt embedding arrays when --release-after-run is set.")
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser()
    if not args.skip_bootstrap:
        requested_dtypes = parse_convert_dtypes(args.convert_dtypes)
        runtime_dtypes = dtypes_for_runtime(args.weights_dtype, args.decoder_backend, args.decoder_weights_dtype)
        convert_dtypes = tuple(dict.fromkeys([*requested_dtypes, *runtime_dtypes]))
        config = BootstrapConfig(
            model_dir=model_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            include_t2v=args.task == "T2V",
            auto_download=args.auto_download,
            auto_convert=args.auto_convert,
            convert_dtypes=convert_dtypes,
        )
        try:
            ensure_runtime_ready(config, progress=lambda msg: print(f"[xGRN] {msg}", flush=True))
        except ModelBootstrapError as exc:
            print(f"[xGRN] Startup blocked:\n{exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    result = generate_mac(
        task=args.task,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        pn=args.pn,
        steps=args.steps,
        guidance=args.guidance,
        temperature=args.temperature,
        duration=args.duration,
        model_dir=model_dir,
        text_dtype=args.text_dtype,
        text_cache_dtype=args.text_cache_dtype,
        weights_dtype=args.weights_dtype,
        compute_dtype=args.compute_dtype,
        decoder_backend=args.decoder_backend,
        decoder_weights_dtype=args.decoder_weights_dtype,
        decoder_compute_dtype=args.decoder_compute_dtype,
        compile_blocks=args.compile_blocks,
        compile_visual_pass=args.compile_visual_pass,
        compile_cfg_logits=args.compile_cfg_logits,
        compile_refinement_update=args.compile_refinement_update,
        linear_quantization=args.linear_quantization,
        fuse_mlp_gate_up=args.fuse_mlp_gate_up,
        fuse_swiglu_metal=args.fuse_swiglu_metal,
        fuse_rope_metal=args.fuse_rope_metal,
        fuse_residual_norm_metal=args.fuse_residual_norm_metal,
        fuse_qkv_metal=args.fuse_qkv_metal,
        fuse_qkv_concat=args.fuse_qkv_concat,
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
        progress=lambda msg: print(msg, flush=True),
    )
    print(
        {
            "elapsed_sec": round(result.elapsed_sec, 3),
            "timings": {k: round(v, 3) for k, v in result.timings.items()},
            "raw_shape": result.raw_shape,
            "video": str(result.video) if result.video else None,
        }
    )


if __name__ == "__main__":
    main()
