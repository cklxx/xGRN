from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

from xgrn_mlx.bootstrap import (
    BootstrapConfig,
    ModelBootstrapError,
    convert_dtypes_from_env,
    env_bool,
    ensure_runtime_ready,
    model_dir_from_env,
    parse_convert_dtypes,
    repo_id_from_env,
    revision_from_env,
)
from xgrn_mlx.run import generate_mac


NEGATIVE_PROMPT = (
    "ugly, blurry, low-resolution, low-detail, low-quality, noisy, artifacts, "
    "text, watermark, logo, bad composition, deformed, mutated"
)

APP_MODEL_DIR = model_dir_from_env()


def run(
    task: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    resolution: str,
    steps: int,
    guidance_scale: float,
    temperature: float,
    text_dtype: str,
    text_cache_dtype: str,
    weights_dtype: str,
    compute_dtype: str,
    decoder_backend: str,
    sampling_mode: str,
    mask_schedule: str,
    fuse_mlp_gate_up: bool,
    fuse_swiglu_metal: bool,
    stack_cfg_cache: bool,
    min_change_frac: float,
    track_token_confidence: bool,
    precompute_pt_embed: bool,
    aspect: str,
    duration: float,
    capture_interval: int,
    progress=gr.Progress(track_tqdm=True),
):
    h_div_w = {
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
    }[aspect]

    def report(msg: str) -> None:
        progress(0, desc=msg)

    result = generate_mac(
        task=task,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        pn=resolution,
        steps=int(steps),
        guidance=float(guidance_scale),
        temperature=float(temperature),
        text_dtype=text_dtype,
        text_cache_dtype=text_cache_dtype,
        weights_dtype=weights_dtype,
        compute_dtype=compute_dtype,
        decoder_backend=decoder_backend,
        sampling_mode=sampling_mode,
        mask_schedule=mask_schedule,
        fuse_mlp_gate_up=bool(fuse_mlp_gate_up),
        fuse_swiglu_metal=bool(fuse_swiglu_metal),
        stack_cfg_cache=bool(stack_cfg_cache),
        min_change_frac=float(min_change_frac),
        track_token_confidence=bool(track_token_confidence),
        precompute_pt_embed=bool(precompute_pt_embed),
        h_div_w=float(h_div_w),
        duration=float(duration),
        capture_interval=int(capture_interval),
        model_dir=APP_MODEL_DIR,
        progress=report,
    )
    df = pd.DataFrame(result.stats)
    if not df.empty:
        df = df[[c for c in ["step", "cur_pt", "next_pt", "entropy"] if c in df.columns]]
    summary = {
        "device": f"bf16 UMT5 text + MLX GRN transformer + {decoder_backend} HBQ decoder",
        "task": task,
        "text_dtype": text_dtype,
        "text_cache_dtype": text_cache_dtype,
        "weights_dtype": weights_dtype,
        "compute_dtype": compute_dtype,
        "decoder_backend": decoder_backend,
        "sampling_mode": sampling_mode,
        "mask_schedule": mask_schedule,
        "fuse_mlp_gate_up": bool(fuse_mlp_gate_up),
        "fuse_swiglu_metal": bool(fuse_swiglu_metal),
        "stack_cfg_cache": bool(stack_cfg_cache),
        "min_change_frac": float(min_change_frac),
        "track_token_confidence": bool(track_token_confidence),
        "precompute_pt_embed": bool(precompute_pt_embed),
        "elapsed_sec": round(result.elapsed_sec, 2),
        "timings": {k: round(v, 3) for k, v in result.timings.items()},
        "raw_shape": result.raw_shape,
        "latest_image": "outputs/latest_t2i.png" if task == "T2I" else "outputs/latest_t2v_first_frame.png",
        "latest_video": str(result.video) if result.video else None,
        "latest_refinement_gif": str(result.refinement_gif) if result.refinement_gif else None,
        "stats_csv": "outputs/refinement_stats.csv",
        "model_dir": str(APP_MODEL_DIR),
    }
    frames = result.refinement_frames if result.refinement_frames else []
    return result.image, str(result.video) if result.video else None, frames, df, json.dumps(summary, indent=2)


with gr.Blocks(title="xGRN Metal") as demo:
    gr.Markdown("# xGRN Metal")
    with gr.Row():
        with gr.Column(scale=2):
            task = gr.Dropdown(label="Task", choices=["T2I", "T2V"], value="T2I")
            prompt = gr.Textbox(
                label="Prompt",
                value="A realistic photo of an orange tabby cat sitting on a windowsill, fluffy fur, green eyes, soft daylight, natural indoor background, sharp focus, warm but realistic colors",
                lines=4,
            )
            negative_prompt = gr.Textbox(label="Negative prompt", value=NEGATIVE_PROMPT, lines=3)
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                resolution = gr.Dropdown(
                    label="Resolution preset",
                    choices=["0.06M", "0.25M", "0.41M", "1M"],
                    value="0.25M",
                )
                aspect = gr.Dropdown(label="Aspect", choices=["1:1", "4:3", "3:4", "16:9", "9:16"], value="1:1")
            with gr.Row():
                steps = gr.Slider(label="Refinement steps", minimum=4, maximum=250, value=50, step=1)
                guidance_scale = gr.Slider(label="Guidance", minimum=1.0, maximum=7.0, value=3.0, step=0.1)
                temperature = gr.Slider(label="Temperature", minimum=0.6, maximum=1.6, value=1.1, step=0.05)
                text_dtype = gr.Dropdown(label="Text dtype", choices=["bf16", "fp16", "fp32"], value="bf16")
                text_cache_dtype = gr.Dropdown(label="Text cache", choices=["fp32", "fp16"], value="fp32")
                weights_dtype = gr.Dropdown(label="GRN weights", choices=["fp32", "fp16", "auto"], value="fp32")
                compute_dtype = gr.Dropdown(label="GRN compute", choices=["bf16", "fp32", "fp16"], value="bf16")
                decoder_backend = gr.Dropdown(label="HBQ decoder", choices=["native", "mps"], value="native")
                sampling_mode = gr.Dropdown(label="Sampling", choices=["categorical", "binary", "argmax"], value="categorical")
                mask_schedule = gr.Dropdown(label="Mask schedule", choices=["random", "dus"], value="random")
                fuse_mlp_gate_up = gr.Checkbox(label="Fuse MLP gate/up", value=False)
                fuse_swiglu_metal = gr.Checkbox(label="Fuse SwiGLU Metal", value=False)
                stack_cfg_cache = gr.Checkbox(label="Stack CFG cache", value=False)
                track_token_confidence = gr.Checkbox(label="Track token confidence", value=False)
                precompute_pt_embed = gr.Checkbox(label="Precompute pt embed", value=False)
                min_change_frac = gr.Slider(
                    label="Early stop threshold",
                    minimum=0.0,
                    maximum=0.05,
                    value=0.0,
                    step=0.001,
                )
            duration = gr.Slider(label="T2V duration seconds", minimum=0.25, maximum=2.0, value=0.25, step=0.25)
            capture_interval = gr.Slider(
                label="Capture frame every N steps (0 = off)",
                minimum=0,
                maximum=25,
                value=0,
                step=1,
            )
            button = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            image = gr.Image(label="Generated image", type="pil")
            video = gr.Video(label="Generated video")
            gallery = gr.Gallery(label="Refinement frames", columns=4, height=280)
            stats = gr.Dataframe(label="Refinement metrics")
            summary = gr.Code(label="Run summary", language="json")

    button.click(
        run,
        inputs=[task, prompt, negative_prompt, seed, resolution, steps, guidance_scale, temperature, text_dtype, text_cache_dtype, weights_dtype, compute_dtype, decoder_backend, sampling_mode, mask_schedule, fuse_mlp_gate_up, fuse_swiglu_metal, stack_cfg_cache, min_change_frac, track_token_confidence, precompute_pt_embed, aspect, duration, capture_interval],
        outputs=[image, video, gallery, stats, summary],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start the xGRN Gradio app. On first run, xGRN checks the local model cache, "
            "downloads missing GRN weights from HuggingFace, and converts them to MLX artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-name", default=os.environ.get("XGRN_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("XGRN_SERVER_PORT", "7860")))
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
    parser.add_argument("--skip-bootstrap", action="store_true", help="Start the UI without checking model assets.")
    parser.add_argument("--bootstrap-only", action="store_true", help="Check/download/convert model assets, then exit before launching Gradio.")
    args = parser.parse_args()

    global APP_MODEL_DIR
    APP_MODEL_DIR = args.model_dir.expanduser()

    if not args.skip_bootstrap:
        config = BootstrapConfig(
            model_dir=APP_MODEL_DIR,
            repo_id=args.repo_id,
            revision=args.revision,
            include_t2v=True,
            auto_download=args.auto_download,
            auto_convert=args.auto_convert,
            convert_dtypes=parse_convert_dtypes(args.convert_dtypes),
        )
        try:
            ensure_runtime_ready(config, progress=lambda msg: print(f"[xGRN] {msg}", flush=True))
        except ModelBootstrapError as exc:
            print(f"[xGRN] Startup blocked:\n{exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    if args.bootstrap_only:
        print(f"[xGRN] bootstrap complete: {APP_MODEL_DIR}", flush=True)
        return
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
