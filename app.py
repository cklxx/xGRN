from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
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


# ─── Defaults ────────────────────────────────────────────────────────────────
NEGATIVE_PROMPT = (
    "ugly, blurry, low-resolution, low-detail, low-quality, noisy, artifacts, "
    "text, watermark, logo, bad composition, deformed, mutated"
)

QUALITY_PRESETS: dict[str, dict] = {
    "Fast": {"pn": "0.06M", "steps": 24, "guidance": 3.0, "label": "⚡ Fast", "hint": "≈1s warm"},
    "Balanced": {"pn": "0.25M", "steps": 50, "guidance": 3.0, "label": "◆ Balanced", "hint": "≈4s warm"},
    "Quality": {"pn": "0.41M", "steps": 100, "guidance": 3.0, "label": "✦ Quality", "hint": "≈12s warm"},
}

ASPECT_RATIOS = {
    "1:1 Square": 1.0,
    "4:3 Landscape": 4 / 3,
    "3:4 Portrait": 3 / 4,
    "16:9 Wide": 16 / 9,
    "9:16 Vertical": 9 / 16,
}

EXAMPLE_PROMPTS = [
    "A realistic photo of an orange tabby cat sitting on a windowsill, fluffy fur, green eyes, soft daylight, natural indoor background, sharp focus, warm but realistic colors",
    "A breakfast table with fresh croissants, espresso, and a vase of yellow tulips, morning light, shallow depth of field",
    "A cinematic aerial view of a coastal cliff at sunset, waves crashing, warm orange sky, ultra detailed",
    "A cozy reading nook with a window, raining outside, warm lamp light, autumn leaves visible through the glass",
    "A futuristic tokyo skyline at night, neon reflections on wet streets, cyberpunk atmosphere, ultra detailed",
]

OUTPUT_DIR = Path("outputs")
APP_MODEL_DIR = model_dir_from_env()


# ─── Server bootstrap helpers (unchanged behavior) ───────────────────────────

def _port_bind_error(server_name: str, server_port: int) -> str | None:
    host = "" if server_name in {"0.0.0.0", "::"} else server_name
    try:
        infos = socket.getaddrinfo(host, server_port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return f"invalid --server-name {server_name!r}: {exc}"
    errors: list[OSError] = []
    for family, socktype, proto, _, sockaddr in infos:
        with socket.socket(family, socktype, proto) as sock:
            try:
                sock.bind(sockaddr)
                return None
            except OSError as exc:
                errors.append(exc)
    if errors:
        return str(errors[0])
    return "no socket address was available"


def _next_available_port(server_name: str, start_port: int) -> int:
    for port in range(start_port + 1, start_port + 21):
        if _port_bind_error(server_name, port) is None:
            return port
    return start_port + 1


def _ensure_server_port_available(server_name: str, server_port: int) -> None:
    bind_error = _port_bind_error(server_name, server_port)
    if bind_error is None:
        return
    suggested_port = _next_available_port(server_name, server_port)
    print(
        "[xGRN] Startup blocked:\n"
        f"Port {server_port} is not available on {server_name}: {bind_error}\n\n"
        "Fix:\n"
        f"  uv run xgrn-app --server-port {suggested_port}\n"
        f"  # or set XGRN_SERVER_PORT={suggested_port}",
        file=sys.stderr,
    )
    raise SystemExit(2)


# ─── Generate ────────────────────────────────────────────────────────────────

GALLERY_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4", "*.gif")


def _gallery_items() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    files: list[Path] = []
    for ext in GALLERY_EXTS:
        files.extend(OUTPUT_DIR.glob(ext))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:80]]


def _random_seed() -> int:
    return random.SystemRandom().randint(0, 2**31 - 1)


def run(
    mode: str,
    prompt: str,
    quality_key: str,
    aspect: str,
    seed: int,
    duration: float,
    use_preset: bool,
    custom_steps: int,
    custom_guidance: float,
    temperature: float,
    negative_prompt: str,
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
    capture_interval: int,
    progress=gr.Progress(track_tqdm=True),
):
    task = "T2V" if mode.startswith("Video") else "T2I"
    preset = QUALITY_PRESETS[quality_key]
    pn = preset["pn"]
    steps = preset["steps"] if use_preset else int(custom_steps)
    guidance = preset["guidance"] if use_preset else float(custom_guidance)

    h_div_w = ASPECT_RATIOS[aspect]

    def report(msg: str) -> None:
        progress(0, desc=msg)

    started = time.perf_counter()
    result = generate_mac(
        task=task,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        pn=pn,
        steps=steps,
        guidance=guidance,
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
    elapsed = result.elapsed_sec or (time.perf_counter() - started)

    df = pd.DataFrame(result.stats)
    if not df.empty:
        df = df[[c for c in ["step", "cur_pt", "next_pt", "entropy"] if c in df.columns]]

    summary = {
        "task": task,
        "preset": quality_key,
        "pn": pn,
        "steps": steps,
        "guidance": guidance,
        "elapsed_sec": round(elapsed, 2),
        "timings": {k: round(v, 3) for k, v in result.timings.items()},
        "raw_shape": result.raw_shape,
        "output": (
            str(result.video) if (task == "T2V" and result.video) else "outputs/latest_t2i.png"
        ),
    }

    caption = (
        f"<div class='caption-row'>"
        f"<span class='pill ok'>{elapsed:.1f}s</span>"
        f"<span class='pill'>{pn} · {steps} steps</span>"
        f"<span class='pill'>seed {int(seed)}</span>"
        f"<span class='pill'>{task}</span>"
        f"</div>"
    )

    image_out = result.image if task == "T2I" else None
    video_out = str(result.video) if (task == "T2V" and result.video) else None
    frames = result.refinement_frames or []

    return (
        gr.update(visible=task == "T2I", value=image_out),
        gr.update(visible=task == "T2V", value=video_out),
        caption,
        frames,
        df,
        json.dumps(summary, indent=2),
        _gallery_items(),
    )


# ─── UI ──────────────────────────────────────────────────────────────────────

# ─── Palette ─────────────────────────────────────────────────────────────────
# Warm cream + terra-cotta + dusty teal. Anthropic-style paper aesthetic.
#   bg cream     #F4F2EE   card white   #FFFFFF   sub-card   #FAF8F4
#   border       #E5E1D8   ink          #1F1F1F   muted      #6B6660
#   primary      #C96442   primary hov  #B5573A   primary 8% #FDEFE9
#   secondary    #5E8B7E   secondary 8% #EEF3F1
#   ink-pre      #2A2520   (only dark surface — code blocks, for emphasis)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fdf5f1",
        c100="#fbe6dd",
        c200="#f5cab6",
        c300="#eaa386",
        c400="#dd8260",
        c500="#c96442",
        c600="#b5573a",
        c700="#94462f",
        c800="#73362a",
        c900="#5b2d24",
        c950="#3d1d18",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f3f7f5",
        c100="#e1eae6",
        c200="#c4d4cd",
        c300="#a3bbb1",
        c400="#82a193",
        c500="#5e8b7e",
        c600="#4d7367",
        c700="#3f5d54",
        c800="#324a43",
        c900="#283b35",
        c950="#1d2925",
    ),
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size="md",
    spacing_size="md",
).set(
    body_background_fill="#F4F2EE",
    body_background_fill_dark="#F4F2EE",
    background_fill_primary="#FAF8F4",
    background_fill_primary_dark="#FAF8F4",
    background_fill_secondary="#ECEAE4",
    background_fill_secondary_dark="#ECEAE4",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E5E1D8",
    block_border_color_dark="#E5E1D8",
    block_border_width="1px",
    border_color_primary="#E5E1D8",
    border_color_primary_dark="#E5E1D8",
    body_text_color="#1F1F1F",
    body_text_color_dark="#1F1F1F",
    body_text_color_subdued="#6B6660",
    body_text_color_subdued_dark="#6B6660",
    button_primary_background_fill="#C96442",
    button_primary_background_fill_dark="#C96442",
    button_primary_background_fill_hover="#B5573A",
    button_primary_background_fill_hover_dark="#B5573A",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#ECEAE4",
    button_secondary_background_fill_dark="#ECEAE4",
    button_secondary_background_fill_hover="#E0DCD2",
    button_secondary_background_fill_hover_dark="#E0DCD2",
    button_secondary_text_color="#1F1F1F",
    button_secondary_text_color_dark="#1F1F1F",
    block_label_text_color="#6B6660",
    block_label_text_color_dark="#6B6660",
    block_title_text_weight="600",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E5E1D8",
    input_border_color_dark="#E5E1D8",
    input_border_color_focus="#C96442",
    input_border_color_focus_dark="#C96442",
)

CSS = """
:root { color-scheme: light; }
html, body, gradio-app { background: #F4F2EE !important; }
gradio-app {
  background:
    radial-gradient(900px 500px at 92% -8%, rgba(201,100,66,0.045), transparent 60%),
    radial-gradient(700px 400px at -4% 100%, rgba(94,139,126,0.04), transparent 70%),
    #F4F2EE !important;
}
.gradio-container { max-width: 1180px !important; padding: 36px 24px 64px !important; }
footer { display: none !important; }

/* Hero ─────────────────────────────────────────────────────────────────── */
#hero { padding: 18px 4px 16px; }
#hero .brand { display: flex; align-items: center; gap: 16px; }
#hero .mark {
  width: 44px; height: 44px; border-radius: 10px;
  background: #C96442;
  box-shadow: 0 1px 0 rgba(255,255,255,0.4) inset, 0 2px 6px rgba(201,100,66,0.22);
  display: grid; place-items: center; color: #FFFFFF; font-weight: 600; font-size: 19px;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: -0.02em;
}
#hero h1 {
  font-size: 1.95rem; font-weight: 600; letter-spacing: -0.025em; margin: 0;
  color: #1F1F1F;
}
#hero .tag { color: #6B6660; font-size: 0.95rem; margin-top: 2px; }
#hero .stats { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
#hero .stat {
  padding: 6px 12px; border-radius: 6px;
  background: #FAF8F4; border: 1px solid #E5E1D8;
  color: #6B6660; font-size: 0.8rem; letter-spacing: 0;
}
#hero .stat b { color: #C96442; font-family: "IBM Plex Mono", monospace; font-weight: 600; font-style: normal; }
#hero .stat.violet b { color: #5E8B7E; }

/* Cards ───────────────────────────────────────────────────────────────── */
.card {
  background: #FFFFFF !important;
  border: 1px solid #E5E1D8 !important;
  border-radius: 12px !important;
  padding: 20px !important;
  box-shadow: 0 1px 2px rgba(28,25,23,0.03), 0 8px 24px -16px rgba(28,25,23,0.06);
}
.card.tight { padding: 16px !important; }

/* Prompt — make it the hero of the form */
#prompt-card textarea {
  font-size: 1.02rem !important;
  line-height: 1.55 !important;
  background: transparent !important;
  border: none !important;
  padding: 4px 0 !important;
  min-height: 92px !important;
  color: #1F1F1F !important;
  resize: vertical;
}
#prompt-card textarea::placeholder { color: #A39E96 !important; }
#prompt-card label {
  color: #6B6660 !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
}

/* Block label inner spans — Gradio's default tints them with primary, kill it */
span.svelte-jdcl7l:not(.sr-only):not(.hide),
[data-testid="block-info"]:not(.sr-only):not(.hide) {
  background: transparent !important;
  color: #6B6660 !important;
  padding: 0 0 4px !important;
  margin: 0 !important;
  font-weight: 500 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.04em !important;
  border-radius: 0 !important;
  display: inline-block;
}
/* Segmented label inner spans should NOT inherit the above */
fieldset.segmented span,
fieldset.segmented [data-testid="block-info"] {
  font-size: inherit !important;
  letter-spacing: 0 !important;
  padding: 0 !important;
}

/* Segmented (Mode + Quality) ──────────────────────────────────────────── */
fieldset.segmented {
  display: flex !important;
  flex-direction: column;
  gap: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: unset !important;
}
fieldset.segmented [data-testid="status-tracker"] { display: none !important; }
fieldset.segmented [data-testid="block-info"]:not(.hide):not(.sr-only) {
  color: #6B6660 !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
  margin-bottom: 6px !important;
  background: transparent !important;
}
fieldset.segmented .wrap.svelte-e4x47i,
fieldset.segmented > div.wrap {
  display: flex !important;
  flex-wrap: wrap;
  gap: 4px !important;
  background: #ECEAE4 !important;
  border-radius: 8px !important;
  padding: 4px !important;
  border: none !important;
}
fieldset.segmented label {
  flex: 1 1 auto;
  display: inline-flex !important;
  align-items: center; justify-content: center;
  gap: 6px;
  border-radius: 6px !important;
  padding: 8px 14px !important;
  background: transparent !important;
  border: none !important;
  color: #6B6660 !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  cursor: pointer;
  transition: background .15s ease, color .15s ease, box-shadow .15s ease;
  margin: 0 !important;
}
fieldset.segmented label:hover {
  background: rgba(255,255,255,0.7) !important;
  color: #1F1F1F !important;
}
fieldset.segmented label.selected,
fieldset.segmented label:has(input:checked) {
  background: #FFFFFF !important;
  color: #1F1F1F !important;
  box-shadow: 0 1px 2px rgba(28,25,23,0.06), 0 0 0 1px rgba(201,100,66,0.25);
}
fieldset.segmented input[type="radio"] { display: none !important; }
fieldset.segmented label > span { color: inherit !important; font-weight: inherit !important; }

/* Generate ─────────────────────────────────────────────────────────────── */
#generate-btn { margin-top: 4px; }
#generate-btn button {
  height: 52px !important;
  font-size: 1.0rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
  border-radius: 10px !important;
  border: none !important;
  background: #C96442 !important;
  color: #FFFFFF !important;
  box-shadow: 0 1px 0 rgba(255,255,255,0.18) inset, 0 2px 6px rgba(201,100,66,0.18) !important;
  transition: background .15s ease, transform .08s ease, box-shadow .15s ease !important;
}
#generate-btn button:hover {
  background: #B5573A !important;
  transform: translateY(-1px);
  box-shadow: 0 1px 0 rgba(255,255,255,0.18) inset, 0 4px 12px rgba(201,100,66,0.26) !important;
}
#generate-btn button:active { transform: translateY(0); }

/* Dice */
#dice-btn button {
  height: 100%; min-height: 42px;
  border-radius: 8px !important; font-size: 1.05rem !important;
  background: #ECEAE4 !important;
  color: #1F1F1F !important;
  border: 1px solid #E5E1D8 !important;
  box-shadow: none !important;
}
#dice-btn button:hover { background: #E0DCD2 !important; }

/* Preview canvas ──────────────────────────────────────────────────────── */
#preview-card { min-height: 520px; display: flex; flex-direction: column; }
#preview-card .image-container, #preview-card .video-container {
  background: #FAF8F4 !important;
  border-radius: 8px !important;
  overflow: hidden;
  border: 1px solid #E5E1D8;
}
#preview-card img, #preview-card video { border-radius: 6px; }
.empty-canvas {
  flex: 1; min-height: 460px; border-radius: 8px;
  background: #FAF8F4;
  display: grid; place-items: center; color: #B0AAA0; font-size: 0.9rem;
  border: 1px dashed #E5E1D8;
}

/* Caption pills */
.caption-row { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; margin-top: 12px; }
.pill {
  padding: 4px 10px; border-radius: 6px;
  background: #ECEAE4; border: 1px solid #E5E1D8;
  color: #6B6660; font-size: 0.76rem; font-family: "IBM Plex Mono", monospace;
}
.pill.ok { color: #FFFFFF; background: #C96442; border-color: #C96442; }

/* Tabs ────────────────────────────────────────────────────────────────── */
.tab-nav {
  border-bottom: 1px solid #E5E1D8 !important;
  gap: 0 !important;
  background: transparent !important;
  margin-bottom: 18px !important;
}
.tab-nav button {
  color: #6B6660 !important;
  font-weight: 500 !important;
  padding: 10px 16px !important;
  border-radius: 0 !important;
  background: transparent !important;
  border: none !important;
  margin: 0 !important;
}
.tab-nav button:hover { color: #1F1F1F !important; }
.tab-nav button.selected {
  color: #1F1F1F !important;
  background: transparent !important;
  box-shadow: inset 0 -2px 0 #C96442 !important;
}

/* Accordion */
.label-wrap { color: #6B6660 !important; }
button.label-wrap, .accordion > button { font-weight: 500 !important; color: #1F1F1F !important; }

/* Examples ────────────────────────────────────────────────────────────── */
button.gallery-item, .gallery-item {
  background: #FAF8F4 !important;
  border: 1px solid #E5E1D8 !important;
  color: #1F1F1F !important;
  border-radius: 8px !important;
  text-align: left !important;
  padding: 10px 12px !important;
  margin: 0 !important;
  transition: all .15s ease;
  box-shadow: none !important;
}
button.gallery-item:hover, .gallery-item:hover {
  background: #FFFFFF !important;
  border-color: #C96442 !important;
  color: #1F1F1F !important;
}
button.gallery-item .gallery, .gallery-item .gallery, .gallery-item span {
  color: inherit !important;
  font-size: 0.85rem !important;
  white-space: normal !important;
  --local-text-width: 100% !important;
}
#examples > .label-wrap, #examples label.label-wrap {
  color: #6B6660 !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600 !important;
}

/* Gallery thumbnails ──────────────────────────────────────────────────── */
.grid-wrap { gap: 10px !important; }
.thumbnail-item {
  border-radius: 8px !important;
  overflow: hidden;
  border: 1px solid #E5E1D8 !important;
  transition: all .15s ease;
}
.thumbnail-item:hover {
  transform: translateY(-2px);
  border-color: #C96442 !important;
  box-shadow: 0 4px 12px rgba(28,25,23,0.08);
}

/* Markdown ───────────────────────────────────────────────────────────── */
.prose code, .markdown code {
  background: #ECEAE4 !important;
  color: #C96442 !important;
  padding: 1px 6px !important;
  border-radius: 4px !important;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 0.86em !important;
  border: 1px solid #DBD7CE !important;
}
.prose pre, .markdown pre {
  background: #2A2520 !important;
  border: 1px solid #2A2520 !important;
  border-radius: 8px !important;
  padding: 14px !important;
  overflow-x: auto;
}
.prose pre code, .markdown pre code {
  background: transparent !important; border: none !important;
  padding: 0 !important; color: #F4F2EE !important;
}
.prose table, .markdown table {
  border-collapse: collapse !important;
  border: 1px solid #E5E1D8 !important;
  border-radius: 8px !important; overflow: hidden;
  margin: 12px 0;
  background: #FFFFFF !important;
}
.prose td, .prose th, .markdown td, .markdown th {
  border: 1px solid #E5E1D8 !important;
  padding: 9px 14px !important;
  background: #FFFFFF !important;
  color: #1F1F1F !important;
}
.prose th, .markdown th {
  background: #FAF8F4 !important;
  color: #1F1F1F !important; font-weight: 600;
}
.prose h1, .prose h2, .prose h3, .markdown h1, .markdown h2, .markdown h3 {
  color: #1F1F1F !important; letter-spacing: -0.01em;
}
.prose p, .markdown p, .prose li, .markdown li { color: #1F1F1F !important; }

/* Notice banner — dusty teal */
.notice {
  padding: 10px 14px; border-radius: 8px; font-size: 0.85rem;
  background: #EEF3F1;
  border: 1px solid rgba(94,139,126,0.25);
  color: #324A43;
  margin: 8px 0 14px;
}
.notice b { color: #1F2A26; }

/* Inputs polish — softer focus */
input, textarea, select {
  background: #FFFFFF !important;
  color: #1F1F1F !important;
  border-color: #E5E1D8 !important;
}
input:focus, textarea:focus, select:focus {
  border-color: #C96442 !important;
  box-shadow: 0 0 0 3px rgba(201,100,66,0.12) !important;
  outline: none !important;
}

/* Sliders — terra accent */
input[type="range"]::-webkit-slider-thumb { background: #C96442 !important; }
input[type="range"]::-moz-range-thumb { background: #C96442 !important; }

/* Code block (run summary JSON) */
.code-wrap, .code-wrap pre, .code-wrap code {
  background: #2A2520 !important;
  color: #F4F2EE !important;
  border-color: #2A2520 !important;
}
"""

HERO_HTML = """
<div id='hero'>
  <div class='brand'>
    <div class='mark'>x</div>
    <div>
      <h1>xGRN</h1>
      <div class='tag'>GRN T2I &middot; T2V on Apple Metal &middot; native MLX runtime</div>
    </div>
  </div>
  <div class='stats'>
    <div class='stat'><b>102&times;</b> faster vs PyTorch/MPS (warm)</div>
    <div class='stat violet'><b>25&times;</b> faster cold start</div>
    <div class='stat'>bf16 UMT5 &middot; MLX GRN &middot; native HBQ</div>
  </div>
</div>
"""

ABOUT_MD = """
### What this is

A Mac-specialised runtime for the official **GRN** T2I/T2V models. Text encoding is bf16 UMT5,
the GRN transformer + refinement loop run in **MLX**, and HBQ decode uses a native MLX decoder.
Up to **102×** faster than stock PyTorch/MPS on the same Mac.

### How to use

1. Type a prompt (or pick an example).
2. Choose **Image** or **Video**, pick a **Quality** preset, choose an aspect.
3. Hit **Generate**.

That's it. Power-user knobs (dtypes, kernel fusion, sampling mode) live under **Advanced**.

### First-run download

The first time you hit **Generate**, xGRN downloads the official GRN HuggingFace snapshot
into `models/GRN/` and creates MLX artifacts. Plan for several GB free disk and a few
minutes on a fast network. Subsequent launches reuse the cache.

You can also pre-fetch from the terminal:

```bash
uv run xgrn-download --model-dir models/GRN
```

### Outputs

| File | Description |
|---|---|
| `outputs/latest_t2i.png` | Most recent image |
| `outputs/latest_t2v.mp4` | Most recent video |
| `outputs/latest_t2v_first_frame.png` | First frame preview |
| `outputs/refinement_stats.csv` | Per-step metrics |
"""

# Build UI
with gr.Blocks(title="xGRN — Apple Metal", analytics_enabled=False) as demo:
    gr.HTML(HERO_HTML)

    with gr.Tabs():
        # ── Create tab ─────────────────────────────────────────────────────
        with gr.Tab("Create"):
            with gr.Row(equal_height=False):
                # Left column — controls
                with gr.Column(scale=5, min_width=420):
                    with gr.Group(elem_classes="card", elem_id="prompt-card"):
                        prompt = gr.Textbox(
                            label="PROMPT",
                            placeholder="A realistic photo of an orange tabby cat on a windowsill…",
                            value=EXAMPLE_PROMPTS[0],
                            lines=4,
                            show_label=True,
                        )
                        gr.Examples(
                            EXAMPLE_PROMPTS,
                            inputs=prompt,
                            label="Try one of these",
                            elem_id="examples",
                        )

                    with gr.Group(elem_classes="card tight"):
                        with gr.Row():
                            mode = gr.Radio(
                                choices=["📷 Image", "🎬 Video"],
                                value="📷 Image",
                                label="Mode",
                                elem_classes="segmented",
                                container=False,
                                scale=1,
                            )
                        with gr.Row():
                            quality = gr.Radio(
                                choices=[(p["label"], k) for k, p in QUALITY_PRESETS.items()],
                                value="Balanced",
                                label="Quality preset",
                                info="Fast ≈ 1 s · Balanced ≈ 4 s · Quality ≈ 12 s (warm)",
                                elem_classes="segmented",
                                container=False,
                                scale=1,
                            )
                        with gr.Row():
                            aspect = gr.Dropdown(
                                label="Aspect ratio",
                                choices=list(ASPECT_RATIOS.keys()),
                                value="1:1 Square",
                                scale=3,
                            )
                            seed = gr.Number(label="Seed", value=42, precision=0, scale=2)
                            dice = gr.Button("🎲", scale=0, min_width=48, elem_id="dice-btn")
                        duration = gr.Slider(
                            label="Video duration (seconds)",
                            minimum=0.25,
                            maximum=2.0,
                            value=0.5,
                            step=0.25,
                            visible=False,
                        )

                    button = gr.Button("Generate", variant="primary", elem_id="generate-btn")

                    with gr.Accordion("Advanced — engineer mode", open=False):
                        gr.HTML(
                            "<div class='notice'>These knobs override the <b>Quality</b> preset and "
                            "are only useful if you know what you're doing. Defaults are correctness-tested.</div>"
                        )
                        use_preset = gr.Checkbox(
                            label="Use preset for steps & guidance (uncheck to override below)",
                            value=True,
                        )
                        with gr.Row():
                            custom_steps = gr.Slider(label="Refinement steps", minimum=4, maximum=250, value=50, step=1)
                            custom_guidance = gr.Slider(label="Guidance", minimum=1.0, maximum=7.0, value=3.0, step=0.1)
                            temperature = gr.Slider(label="Temperature", minimum=0.6, maximum=1.6, value=1.1, step=0.05)
                        negative_prompt = gr.Textbox(label="Negative prompt", value=NEGATIVE_PROMPT, lines=2)
                        with gr.Row():
                            text_dtype = gr.Dropdown(label="Text dtype", choices=["bf16", "fp16", "fp32"], value="bf16")
                            text_cache_dtype = gr.Dropdown(label="Text cache", choices=["fp32", "fp16"], value="fp32")
                            weights_dtype = gr.Dropdown(label="GRN weights", choices=["fp32", "fp16", "auto"], value="fp32")
                            compute_dtype = gr.Dropdown(label="GRN compute", choices=["bf16", "fp32", "fp16"], value="bf16")
                        with gr.Row():
                            decoder_backend = gr.Dropdown(label="HBQ decoder", choices=["native", "mps"], value="native")
                            sampling_mode = gr.Dropdown(label="Sampling", choices=["categorical", "binary", "argmax"], value="categorical")
                            mask_schedule = gr.Dropdown(label="Schedule", choices=["random", "dus"], value="random")
                        with gr.Row():
                            fuse_mlp_gate_up = gr.Checkbox(label="Fuse MLP gate/up", value=False)
                            fuse_swiglu_metal = gr.Checkbox(label="Fuse SwiGLU Metal", value=False)
                            stack_cfg_cache = gr.Checkbox(label="Stack CFG cache", value=False)
                            track_token_confidence = gr.Checkbox(label="Track confidence", value=False)
                            precompute_pt_embed = gr.Checkbox(label="Precompute pt embed", value=False)
                        with gr.Row():
                            min_change_frac = gr.Slider(label="Early-stop threshold", minimum=0.0, maximum=0.05, value=0.0, step=0.001)
                            capture_interval = gr.Slider(label="Capture every N steps (0=off)", minimum=0, maximum=25, value=0, step=1)

                # Right column — preview
                with gr.Column(scale=5, min_width=420):
                    _latest_t2i = OUTPUT_DIR / "latest_t2i.png"
                    _initial_image = str(_latest_t2i) if _latest_t2i.exists() else None
                    _initial_caption = (
                        "<div class='caption-row'>"
                        "<span class='pill'>last result</span>"
                        "<span class='pill'>hit Generate for a new one</span>"
                        "</div>"
                        if _initial_image
                        else "<div class='caption-row'><span class='pill'>Ready</span></div>"
                    )
                    with gr.Group(elem_classes="card", elem_id="preview-card"):
                        image = gr.Image(
                            value=_initial_image,
                            label="",
                            type="pil",
                            show_label=False,
                            container=False,
                            height=520,
                            visible=True,
                        )
                        video = gr.Video(
                            label="",
                            show_label=False,
                            container=False,
                            visible=False,
                            height=520,
                        )
                        caption = gr.HTML(_initial_caption)
                    with gr.Accordion("Run details", open=False):
                        gallery_frames = gr.Gallery(label="Refinement frames", columns=4, height=220)
                        stats = gr.Dataframe(label="Per-step metrics")
                        summary = gr.Code(label="Summary", language="json")

        # ── Gallery tab ────────────────────────────────────────────────────
        with gr.Tab("Gallery"):
            gr.Markdown("Recent files in **`outputs/`**, newest first.")
            with gr.Row():
                refresh = gr.Button("↻ Refresh", scale=0, min_width=110)
            history = gr.Gallery(
                value=_gallery_items(),
                columns=4,
                height=720,
                allow_preview=True,
                show_label=False,
            )

        # ── About tab ──────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    # ── Wiring ─────────────────────────────────────────────────────────────

    def _on_mode_change(m: str):
        is_video = m.startswith("🎬")
        return (
            gr.update(visible=is_video),       # duration
            gr.update(visible=not is_video),   # image
            gr.update(visible=is_video),       # video
        )

    mode.change(_on_mode_change, inputs=mode, outputs=[duration, image, video])
    dice.click(_random_seed, outputs=seed)
    refresh.click(_gallery_items, outputs=history)

    button.click(
        run,
        inputs=[
            mode, prompt, quality, aspect, seed, duration,
            use_preset, custom_steps, custom_guidance, temperature, negative_prompt,
            text_dtype, text_cache_dtype, weights_dtype, compute_dtype,
            decoder_backend, sampling_mode, mask_schedule,
            fuse_mlp_gate_up, fuse_swiglu_metal, stack_cfg_cache,
            min_change_frac, track_token_confidence, precompute_pt_embed,
            capture_interval,
        ],
        outputs=[image, video, caption, gallery_frames, stats, summary, history],
    )


# ─── main() ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start the xGRN Gradio app. On first run, xGRN auto-downloads the GRN weights from "
            "HuggingFace and converts them to MLX artifacts. Re-runs reuse the cache."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-name", default=os.environ.get("XGRN_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("XGRN_SERVER_PORT", "7860")))
    parser.add_argument("--model-dir", type=Path, default=model_dir_from_env(),
                        help="Local GRN model cache directory. Env: XGRN_MODEL_DIR.")
    parser.add_argument("--repo-id", default=repo_id_from_env(),
                        help="HuggingFace model repo id. Env: XGRN_HF_REPO_ID.")
    parser.add_argument("--revision", default=revision_from_env(),
                        help="HuggingFace revision/tag/commit. Env: XGRN_HF_REVISION.")
    parser.add_argument("--convert-dtypes", default=",".join(convert_dtypes_from_env()),
                        help="Comma-separated MLX artifact dtypes to ensure. Env: XGRN_CONVERT_DTYPES.")
    parser.add_argument("--auto-download", dest="auto_download", action=argparse.BooleanOptionalAction,
                        default=env_bool("XGRN_AUTO_DOWNLOAD", True),
                        help="Download missing raw weights; --no-auto-download prints a repair command instead.")
    parser.add_argument("--auto-convert", dest="auto_convert", action=argparse.BooleanOptionalAction,
                        default=env_bool("XGRN_AUTO_CONVERT", True),
                        help="Create missing MLX artifacts; --no-auto-convert prints conversion commands instead.")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Start the UI without checking model assets (UI-preview / dev only).")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="Check/download/convert model assets, then exit before launching Gradio.")
    args = parser.parse_args()

    global APP_MODEL_DIR
    APP_MODEL_DIR = args.model_dir.expanduser()

    if not args.bootstrap_only:
        _ensure_server_port_available(args.server_name, args.server_port)

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
        print(
            f"[xGRN] Checking model cache at {APP_MODEL_DIR} "
            f"(auto-download={args.auto_download}, auto-convert={args.auto_convert})…",
            flush=True,
        )
        try:
            ensure_runtime_ready(config, progress=lambda msg: print(f"[xGRN] {msg}", flush=True))
        except ModelBootstrapError as exc:
            print(f"[xGRN] Startup blocked:\n{exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    if args.bootstrap_only:
        print(f"[xGRN] bootstrap complete: {APP_MODEL_DIR}", flush=True)
        return

    print(
        f"\n  ┌──────────────────────────────────────────────┐\n"
        f"  │  xGRN ready  →  http://{args.server_name}:{args.server_port}        │\n"
        f"  └──────────────────────────────────────────────┘\n",
        flush=True,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
