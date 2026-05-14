from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from huggingface_hub import snapshot_download

from .convert import convert_all
from .text import REFERENCE_GRN


DEFAULT_REPO_ID = "bytedance-research/GRN"
DEFAULT_CONVERT_DTYPES = ("fp16", "fp32")
ROOT = Path(__file__).resolve().parents[1]


class ModelBootstrapError(RuntimeError):
    """Raised when xGRN cannot prepare the model assets needed to run."""


@dataclass(frozen=True)
class BootstrapConfig:
    model_dir: Path
    repo_id: str = DEFAULT_REPO_ID
    revision: str | None = None
    include_t2v: bool = True
    auto_download: bool = True
    auto_convert: bool = True
    convert_dtypes: tuple[str, ...] = DEFAULT_CONVERT_DTYPES


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ModelBootstrapError(
        f"Invalid boolean value for {name}={raw!r}. Use 1/0, true/false, yes/no, or on/off."
    )


def model_dir_from_env(default: Path | str = Path("models/GRN")) -> Path:
    return Path(os.environ.get("XGRN_MODEL_DIR", str(default))).expanduser()


def repo_id_from_env(default: str = DEFAULT_REPO_ID) -> str:
    return os.environ.get("XGRN_HF_REPO_ID", default)


def revision_from_env() -> str | None:
    return os.environ.get("XGRN_HF_REVISION") or None


def convert_dtypes_from_env(default: str = ",".join(DEFAULT_CONVERT_DTYPES)) -> tuple[str, ...]:
    return parse_convert_dtypes(os.environ.get("XGRN_CONVERT_DTYPES", default))


def parse_convert_dtypes(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_CONVERT_DTYPES
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",")]
    else:
        items = [str(item).strip() for item in raw]
    dtypes: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in {"fp16", "fp32"}:
            raise ModelBootstrapError(f"Unsupported MLX conversion dtype {item!r}; use fp16, fp32, or both.")
        if item not in dtypes:
            dtypes.append(item)
    if not dtypes:
        raise ModelBootstrapError("At least one MLX conversion dtype is required.")
    return tuple(dtypes)


def dtypes_for_runtime(
    weights_dtype: str,
    decoder_backend: str = "native",
    decoder_weights_dtype: str = "fp16",
) -> tuple[str, ...]:
    dtypes: set[str] = set()
    if weights_dtype == "auto":
        dtypes.update(DEFAULT_CONVERT_DTYPES)
    else:
        dtypes.add(weights_dtype)
    if decoder_backend == "native":
        dtypes.add(decoder_weights_dtype)
    order = {"fp16": 0, "fp32": 1}
    return tuple(sorted(dtypes, key=lambda item: order[item]))


def _download_patterns(include_t2v: bool) -> list[str]:
    patterns = [
        "GRN_T2I_2B.pth",
        "HBQ_tokenizer_64dim_M4.ckpt",
        "umt5-xxl/**",
        "README.md",
    ]
    if include_t2v:
        patterns.append("GRN_T2V_2B.pth")
    return patterns


def _raw_paths(model_dir: Path, include_t2v: bool) -> list[Path]:
    paths = [
        model_dir / "GRN_T2I_2B.pth",
        model_dir / "HBQ_tokenizer_64dim_M4.ckpt",
        model_dir / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "umt5-xxl" / "umt5-xxl",
    ]
    if include_t2v:
        paths.append(model_dir / "GRN_T2V_2B.pth")
    return paths


def _mlx_dir(model_dir: Path, dtype: str) -> Path:
    return model_dir / ("mlx" if dtype == "fp16" else f"mlx_{dtype}")


def _artifact_paths(model_dir: Path, include_t2v: bool, dtypes: Iterable[str]) -> list[Path]:
    tasks = ["t2i"]
    if include_t2v:
        tasks.append("t2v")
    paths: list[Path] = []
    for dtype in dtypes:
        out_dir = _mlx_dir(model_dir, dtype)
        paths.extend(out_dir / f"grn_{task}_{dtype}.safetensors" for task in tasks)
        paths.append(out_dir / f"hbq_{dtype}.safetensors")
        paths.append(out_dir / "manifest.json")
    return paths


def _missing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _display_paths(paths: Sequence[Path], base: Path) -> str:
    lines = []
    for path in paths:
        try:
            label = path.relative_to(base)
        except ValueError:
            label = path
        lines.append(f"  - {label}")
    return "\n".join(lines)


def _download_command(config: BootstrapConfig) -> str:
    parts = ["uv", "run", "xgrn-download", "--model-dir", str(config.model_dir)]
    if config.repo_id != DEFAULT_REPO_ID:
        parts.extend(["--repo-id", config.repo_id])
    if config.revision:
        parts.extend(["--revision", config.revision])
    if not config.include_t2v:
        parts.append("--t2i-only")
    if config.convert_dtypes:
        parts.extend(["--convert-dtypes", ",".join(config.convert_dtypes)])
    return shlex.join(parts)


def _convert_commands(config: BootstrapConfig) -> str:
    commands = []
    for dtype in config.convert_dtypes:
        out_dir = _mlx_dir(config.model_dir, dtype)
        commands.append(
            shlex.join(
                [
                    "uv",
                    "run",
                    "xgrn-convert",
                    "--model-dir",
                    str(config.model_dir),
                    "--out-dir",
                    str(out_dir),
                    "--dtype",
                    dtype,
                ]
            )
        )
    return "\n".join(f"  {cmd}" for cmd in commands)


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def ensure_reference_repo(progress: Callable[[str], None] | None = None) -> None:
    expected = REFERENCE_GRN / "grn" / "models" / "umt5" / "t5.py"
    if expected.exists():
        _progress(progress, f"official GRN source ready: {REFERENCE_GRN}")
        return
    raise ModelBootstrapError(
        "Official GRN source checkout is required for the UMT5 text encoder.\n"
        f"Expected file: {expected}\n\n"
        "Fix:\n"
        f"  cd {ROOT.parent}\n"
        "  git clone https://github.com/MGenAI/GRN GRN\n\n"
        "The required layout is:\n"
        f"  {ROOT}\n"
        f"  {REFERENCE_GRN}"
    )


def ensure_model_assets(config: BootstrapConfig, progress: Callable[[str], None] | None = None) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    missing_raw = _missing(_raw_paths(config.model_dir, config.include_t2v))
    if missing_raw:
        if not config.auto_download:
            raise ModelBootstrapError(
                "xGRN model cache is incomplete and auto-download is disabled.\n"
                f"Model dir: {config.model_dir}\n"
                f"Missing:\n{_display_paths(missing_raw, config.model_dir)}\n\n"
                "Fix:\n"
                f"  {_download_command(config)}"
            )
        _download_snapshot(config, progress)
        missing_raw = _missing(_raw_paths(config.model_dir, config.include_t2v))
        if missing_raw:
            raise ModelBootstrapError(
                "HuggingFace download finished but required raw model files are still missing.\n"
                f"Model dir: {config.model_dir}\n"
                f"Missing:\n{_display_paths(missing_raw, config.model_dir)}\n\n"
                "Try again manually:\n"
                f"  {_download_command(config)}"
            )
    else:
        _progress(progress, f"raw model cache ready: {config.model_dir}")

    missing_artifacts = _missing(_artifact_paths(config.model_dir, config.include_t2v, config.convert_dtypes))
    if not missing_artifacts:
        _progress(progress, f"MLX artifacts ready: {config.model_dir}")
        return
    if not config.auto_convert:
        raise ModelBootstrapError(
            "xGRN raw weights are present, but required MLX artifacts are missing and auto-convert is disabled.\n"
            f"Model dir: {config.model_dir}\n"
            f"Missing:\n{_display_paths(missing_artifacts, config.model_dir)}\n\n"
            "Fix:\n"
            f"{_convert_commands(config)}"
        )

    for dtype in config.convert_dtypes:
        dtype_missing = _missing(_artifact_paths(config.model_dir, config.include_t2v, (dtype,)))
        if not dtype_missing:
            continue
        out_dir = _mlx_dir(config.model_dir, dtype)
        _progress(progress, f"converting raw weights to MLX {dtype}: {out_dir}")
        convert_all(config.model_dir, out_dir, dtype=dtype)
    missing_artifacts = _missing(_artifact_paths(config.model_dir, config.include_t2v, config.convert_dtypes))
    if missing_artifacts:
        raise ModelBootstrapError(
            "MLX conversion finished but required artifacts are still missing.\n"
            f"Model dir: {config.model_dir}\n"
            f"Missing:\n{_display_paths(missing_artifacts, config.model_dir)}\n\n"
            "Try conversion manually:\n"
            f"{_convert_commands(config)}"
        )
    _progress(progress, f"MLX artifacts ready: {config.model_dir}")


def ensure_runtime_ready(config: BootstrapConfig, progress: Callable[[str], None] | None = None) -> None:
    ensure_reference_repo(progress)
    ensure_model_assets(config, progress)


def _download_snapshot(config: BootstrapConfig, progress: Callable[[str], None] | None) -> None:
    patterns = _download_patterns(config.include_t2v)
    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            _progress(
                progress,
                (
                    f"downloading GRN weights from HuggingFace: repo={config.repo_id} "
                    f"revision={config.revision or 'default'} attempt={attempt}/2"
                ),
            )
            snapshot_download(
                repo_id=config.repo_id,
                repo_type="model",
                revision=config.revision,
                local_dir=str(config.model_dir),
                allow_patterns=patterns,
            )
            return
        except Exception as exc:  # noqa: BLE001 - surface the original hub error with actionable context.
            last_error = exc
            if attempt == 1:
                _progress(progress, f"download failed; retrying once: {exc}")
    raise ModelBootstrapError(
        "Failed to download GRN model assets from HuggingFace after 2 attempts.\n"
        f"Repo: {config.repo_id}\n"
        f"Revision: {config.revision or 'default'}\n"
        f"Model dir: {config.model_dir}\n"
        f"Last error: {last_error}\n\n"
        "Fix:\n"
        "  1. Check network/HuggingFace access and disk space.\n"
        "  2. If the repo requires auth, run `huggingface-cli login` or export HF_TOKEN.\n"
        f"  3. Retry manually: {_download_command(config)}"
    ) from last_error
