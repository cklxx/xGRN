from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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


REPO_ROOT = Path(__file__).resolve().parent
WEB_DIST = REPO_ROOT / "web" / "dist"
OUTPUT_DIR = REPO_ROOT / "outputs"

NEGATIVE_PROMPT = (
    "ugly, blurry, low-resolution, low-detail, low-quality, noisy, artifacts, "
    "text, watermark, logo, bad composition, deformed, mutated"
)

QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "Fast":     {"pn": "0.06M", "steps": 24,  "guidance": 3.0, "label": "⚡ Fast",     "hint": "≈1s warm"},
    "Balanced": {"pn": "0.25M", "steps": 50,  "guidance": 3.0, "label": "◆ Balanced", "hint": "≈4s warm"},
    "Quality":  {"pn": "0.41M", "steps": 100, "guidance": 3.0, "label": "✦ Quality",  "hint": "≈12s warm"},
}

ASPECT_RATIOS: dict[str, float] = {
    "1:1 Square":     1.0,
    "4:3 Landscape":  4 / 3,
    "3:4 Portrait":   3 / 4,
    "16:9 Wide":      16 / 9,
    "9:16 Vertical":  9 / 16,
}

EXAMPLE_PROMPTS = [
    "A realistic photo of an orange tabby cat sitting on a windowsill, fluffy fur, green eyes, soft daylight, natural indoor background, sharp focus, warm but realistic colors",
    "A breakfast table with fresh croissants, espresso, and a vase of yellow tulips, morning light, shallow depth of field",
    "A cinematic aerial view of a coastal cliff at sunset, waves crashing, warm orange sky, ultra detailed",
    "A cozy reading nook with a window, raining outside, warm lamp light, autumn leaves visible through the glass",
    "A futuristic tokyo skyline at night, neon reflections on wet streets, cyberpunk atmosphere, ultra detailed",
]

DEFAULTS = {
    "negative_prompt":  NEGATIVE_PROMPT,
    "text_dtype":       "bf16",
    "text_cache_dtype": "fp32",
    "weights_dtype":    "fp32",
    "compute_dtype":    "bf16",
    "decoder_backend":  "native",
    "sampling_mode":    "categorical",
    "mask_schedule":    "random",
    "temperature":      1.1,
}

GALLERY_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4", "*.gif")


# ─── State ───────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    model_dir: Path
    generation_lock: asyncio.Lock


STATE: AppState | None = None


# ─── Server bootstrap helpers ────────────────────────────────────────────────

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


def _safe_outputs_path(rel: str) -> Path:
    """Resolve a request path to a real file inside outputs/, no traversal."""
    rel = rel.lstrip("/")
    if rel.startswith("outputs/"):
        rel = rel[len("outputs/"):]
    candidate = (OUTPUT_DIR / rel).resolve()
    if not candidate.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="path escapes outputs/")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="not found")
    return candidate


# ─── Pydantic ────────────────────────────────────────────────────────────────

class GenerateBody(BaseModel):
    task: str = Field(..., pattern="^(T2I|T2V)$")
    prompt: str
    quality: str = Field(..., pattern="^(Fast|Balanced|Quality)$")
    aspect: str
    seed: int = 42
    duration: float = 0.5

    use_preset: bool = True
    custom_steps: int = 50
    custom_guidance: float = 3.0
    temperature: float = DEFAULTS["temperature"]
    negative_prompt: str = NEGATIVE_PROMPT

    text_dtype: str = DEFAULTS["text_dtype"]
    text_cache_dtype: str = DEFAULTS["text_cache_dtype"]
    weights_dtype: str = DEFAULTS["weights_dtype"]
    compute_dtype: str = DEFAULTS["compute_dtype"]
    decoder_backend: str = DEFAULTS["decoder_backend"]
    sampling_mode: str = DEFAULTS["sampling_mode"]
    mask_schedule: str = DEFAULTS["mask_schedule"]

    fuse_mlp_gate_up: bool = False
    fuse_swiglu_metal: bool = False
    stack_cfg_cache: bool = False
    track_token_confidence: bool = False
    precompute_pt_embed: bool = False
    min_change_frac: float = 0.0
    capture_interval: int = 0


# ─── App ─────────────────────────────────────────────────────────────────────

def create_app(model_dir: Path, dev: bool = False) -> FastAPI:
    global STATE
    STATE = AppState(model_dir=model_dir, generation_lock=asyncio.Lock())

    app = FastAPI(title="xGRN", docs_url=None, redoc_url=None)

    if dev:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/api/presets")
    def presets() -> dict[str, Any]:
        latest = OUTPUT_DIR / "latest_t2i.png"
        return {
            "qualities": QUALITY_PRESETS,
            "aspects": ASPECT_RATIOS,
            "examples": EXAMPLE_PROMPTS,
            "defaults": DEFAULTS,
            "status": {
                "model_dir": str(model_dir),
                "ready": (model_dir / "mlx_fp32" / "grn_t2i_fp32.safetensors").exists()
                          or (model_dir / "mlx" / "grn_t2i_fp16.safetensors").exists(),
                "latest_image": "outputs/latest_t2i.png" if latest.exists() else None,
            },
        }

    @app.get("/api/history")
    def history() -> list[dict[str, Any]]:
        if not OUTPUT_DIR.exists():
            return []
        files: list[Path] = []
        for ext in GALLERY_EXTS:
            files.extend(OUTPUT_DIR.glob(ext))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for p in files[:120]:
            ext = p.suffix.lower()
            if ext == ".mp4":
                t = "video"
            elif ext == ".gif":
                t = "gif"
            else:
                t = "image"
            items.append({
                "url": f"/api/file/outputs/{p.name}",
                "filename": p.name,
                "mtime": p.stat().st_mtime,
                "type": t,
                "size": p.stat().st_size,
            })
        return items

    @app.get("/api/file/{full_path:path}")
    def serve_output(full_path: str):
        path = _safe_outputs_path(full_path)
        return FileResponse(path)

    @app.post("/api/generate")
    async def generate(body: GenerateBody) -> StreamingResponse:
        """NDJSON stream:
            {"type":"status","msg":"loading prompt embeddings","elapsed_sec":0.30}
            {"type":"status","msg":"running MLX GRN: ...","elapsed_sec":1.40}
            ...
            {"type":"done","result":{...}}
        On failure:
            {"type":"error","message":"..."}
        """
        if STATE is None:
            raise HTTPException(status_code=500, detail="state not initialised")
        if body.aspect not in ASPECT_RATIOS:
            raise HTTPException(status_code=400, detail=f"unknown aspect {body.aspect!r}")

        progress_q: queue.Queue[tuple[str, Any]] = queue.Queue()

        def progress_cb(msg: str) -> None:
            progress_q.put(("status", msg))

        def run_in_thread() -> None:
            try:
                res = _run_generation(body, progress_cb)
                progress_q.put(("done", res))
            except Exception as exc:  # surface to client
                progress_q.put(("error", repr(exc)))

        async def event_stream():
            assert STATE is not None
            # Single-mutex: hold the lock for the whole stream so concurrent
            # requests queue server-side rather than racing the GPU.
            async with STATE.generation_lock:
                start = time.perf_counter()
                t = threading.Thread(target=run_in_thread, daemon=True)
                t.start()
                # Initial event so the client can switch to "running" immediately
                yield _ndjson({"type": "status", "msg": "queued", "elapsed_sec": 0.0})

                while True:
                    try:
                        kind, payload = progress_q.get_nowait()
                    except queue.Empty:
                        if not t.is_alive() and progress_q.empty():
                            yield _ndjson({"type": "error", "message": "worker exited without result"})
                            return
                        await asyncio.sleep(0.08)
                        continue

                    elapsed = round(time.perf_counter() - start, 2)
                    if kind == "status":
                        yield _ndjson({"type": "status", "msg": payload, "elapsed_sec": elapsed})
                    elif kind == "done":
                        yield _ndjson({"type": "done", "result": payload})
                        return
                    elif kind == "error":
                        yield _ndjson({"type": "error", "message": payload})
                        return

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    # ── Static frontend ────────────────────────────────────────────────────
    if WEB_DIST.exists():
        # Serve assets at /assets/* directly
        assets_dir = WEB_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> Response:
            return FileResponse(WEB_DIST / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str) -> Response:
            # Anything that's not /api/* and not /assets/* falls back to index.html
            # (API routes are matched before this thanks to FastAPI's decorator order;
            # this catch-all only fires for client-side routes / favicons / 404s.)
            target = WEB_DIST / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(WEB_DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_dist():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "frontend not built",
                    "fix": "cd web && npm install && npm run build",
                },
            )

    return app


def _ndjson(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, default=str) + "\n").encode("utf-8")


def _run_generation(body: GenerateBody, progress_cb: Callable[[str], None] | None = None) -> dict[str, Any]:
    assert STATE is not None
    preset = QUALITY_PRESETS[body.quality]
    pn = preset["pn"]
    steps = preset["steps"] if body.use_preset else body.custom_steps
    guidance = preset["guidance"] if body.use_preset else body.custom_guidance
    h_div_w = ASPECT_RATIOS[body.aspect]

    started = time.perf_counter()
    result = generate_mac(
        task=body.task,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        seed=int(body.seed),
        pn=pn,
        steps=int(steps),
        guidance=float(guidance),
        temperature=float(body.temperature),
        text_dtype=body.text_dtype,
        text_cache_dtype=body.text_cache_dtype,
        weights_dtype=body.weights_dtype,
        compute_dtype=body.compute_dtype,
        decoder_backend=body.decoder_backend,
        sampling_mode=body.sampling_mode,
        mask_schedule=body.mask_schedule,
        fuse_mlp_gate_up=bool(body.fuse_mlp_gate_up),
        fuse_swiglu_metal=bool(body.fuse_swiglu_metal),
        stack_cfg_cache=bool(body.stack_cfg_cache),
        min_change_frac=float(body.min_change_frac),
        track_token_confidence=bool(body.track_token_confidence),
        precompute_pt_embed=bool(body.precompute_pt_embed),
        h_div_w=float(h_div_w),
        duration=float(body.duration),
        capture_interval=int(body.capture_interval),
        model_dir=STATE.model_dir,
        progress=progress_cb,
    )
    elapsed = result.elapsed_sec or (time.perf_counter() - started)

    image_url: str | None = None
    video_url: str | None = None
    if body.task == "T2I":
        image_url = "/api/file/outputs/latest_t2i.png"
    else:
        first_frame = OUTPUT_DIR / "latest_t2v_first_frame.png"
        if first_frame.exists():
            image_url = "/api/file/outputs/latest_t2v_first_frame.png"
        if result.video:
            video_url = f"/api/file/outputs/{Path(result.video).name}"

    refinement_frames: list[str] = []
    if result.refinement_gif:
        refinement_frames.append(f"/api/file/outputs/{Path(result.refinement_gif).name}")

    summary = {
        "task": body.task,
        "preset": body.quality,
        "pn": pn,
        "steps": int(steps),
        "guidance": float(guidance),
        "seed": int(body.seed),
        "elapsed_sec": round(elapsed, 2),
        "timings": {k: round(v, 3) for k, v in result.timings.items()},
        "raw_shape": list(result.raw_shape),
        "output": video_url or image_url or "",
    }

    caption_parts = [f"{elapsed:.1f}s", f"{pn} · {steps} steps", f"seed {body.seed}", body.task]

    return {
        "ok": True,
        "task": body.task,
        "elapsed_sec": round(elapsed, 2),
        "image_url": image_url,
        "video_url": video_url,
        "caption": " · ".join(caption_parts),
        "summary": summary,
        "stats": result.stats,
        "refinement_frames": refinement_frames,
    }


# ─── main() ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start the xGRN web app. On first run, xGRN auto-downloads the GRN weights "
            "from HuggingFace and converts them to MLX artifacts. Re-runs reuse the cache."
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
                        help="Start the server without checking model assets (UI-preview / dev only).")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="Check/download/convert model assets, then exit before serving.")
    parser.add_argument("--dev", action="store_true",
                        help="Enable CORS for the Vite dev server (http://localhost:5173).")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser()

    if not args.bootstrap_only:
        _ensure_server_port_available(args.server_name, args.server_port)

    if not args.skip_bootstrap:
        config = BootstrapConfig(
            model_dir=model_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            include_t2v=True,
            auto_download=args.auto_download,
            auto_convert=args.auto_convert,
            convert_dtypes=parse_convert_dtypes(args.convert_dtypes),
        )
        print(
            f"[xGRN] Checking model cache at {model_dir} "
            f"(auto-download={args.auto_download}, auto-convert={args.auto_convert})…",
            flush=True,
        )
        try:
            ensure_runtime_ready(config, progress=lambda msg: print(f"[xGRN] {msg}", flush=True))
        except ModelBootstrapError as exc:
            print(f"[xGRN] Startup blocked:\n{exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    if args.bootstrap_only:
        print(f"[xGRN] bootstrap complete: {model_dir}", flush=True)
        return

    if not WEB_DIST.exists():
        print(
            "[xGRN] Frontend bundle not found at web/dist/.\n"
            "       Build it once with:  cd web && npm install && npm run build\n"
            "       Or run the dev server:  cd web && npm run dev   (and pass --dev here)",
            file=sys.stderr,
        )

    app = create_app(model_dir=model_dir, dev=args.dev)
    print(
        f"\n  ┌──────────────────────────────────────────────┐\n"
        f"  │  xGRN ready  →  http://{args.server_name}:{args.server_port}        │\n"
        f"  └──────────────────────────────────────────────┘\n",
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.server_name,
        port=args.server_port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
