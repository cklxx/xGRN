from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.numpy import save_file


REFERENCE_GRN = (Path(__file__).resolve().parents[1] / "../GRN").resolve()
_TEXT_ARRAY_CACHE: dict[str, tuple[mx.array, mx.array]] = {}


def _ensure_reference_importable() -> None:
    if not REFERENCE_GRN.exists():
        raise FileNotFoundError(
            f"Official GRN repo not found at {REFERENCE_GRN}. "
            "It must be a sibling of the xGRN checkout. "
            "See README quickstart: clone GRN alongside xGRN so the layout is ../GRN/."
        )
    root = str(REFERENCE_GRN)
    if root not in sys.path:
        sys.path.insert(0, root)


def _cache_dir(model_dir: Path) -> Path:
    path = model_dir / "mlx" / "text_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_text_ckpt(model_dir: Path) -> Path:
    return model_dir / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth"


def _quant_text_ckpt(model_dir: Path) -> Path:
    return model_dir / ".xgrn_cache" / "umt5-xxl-encoder-fp16.pth"


def _digest(model_dir: Path, prompt: str, negative_prompt: str, text_dtype: str, cache_dtype: str) -> str:
    src = _source_text_ckpt(model_dir)
    stat = src.stat()
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "text_dtype": text_dtype,
        "cache_dtype": cache_dtype,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def ensure_fp16_text_checkpoint(model_dir: Path) -> Path:
    source = _source_text_ckpt(model_dir)
    target = _quant_text_ckpt(model_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = target.with_suffix(".json")
    stat = source.stat()
    expected = {
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "dtype": "float16",
    }
    if target.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()) == expected:
                return target
        except json.JSONDecodeError:
            pass
    state = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    state = {k: v.to(torch.float16) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in state.items()}
    tmp = target.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(target)
    meta.write_text(json.dumps(expected, indent=2))
    del state
    gc.collect()
    return target


def _load_t5_encoder(model_dir: Path, device: torch.device, text_dtype: str):
    _ensure_reference_importable()
    original_current_device = torch.cuda.current_device
    if not torch.cuda.is_available():
        torch.cuda.current_device = lambda: "cpu"  # type: ignore[assignment]
    try:
        from grn.models.umt5.t5 import T5EncoderModel
    finally:
        torch.cuda.current_device = original_current_device
    if text_dtype == "bf16":
        dtype = torch.bfloat16
        checkpoint_path = _source_text_ckpt(model_dir)
    elif text_dtype == "fp16":
        dtype = torch.float16
        checkpoint_path = ensure_fp16_text_checkpoint(model_dir)
    elif text_dtype == "fp32":
        dtype = torch.float32
        checkpoint_path = _source_text_ckpt(model_dir)
    else:
        raise ValueError(f"unsupported text dtype {text_dtype}")
    return T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(model_dir / "umt5-xxl" / "umt5-xxl"),
        enable_fsdp=False,
    )


def _encode_one(text_encoder, text: str, device: torch.device, cache_dtype: str) -> tuple[np.ndarray, list[int]]:
    with torch.no_grad():
        features = text_encoder([text], device)
    lens = [len(item) for item in features]
    dtype = torch.float16 if cache_dtype == "fp16" else torch.float32
    compact = torch.cat(features, dim=0).detach().cpu().to(dtype).numpy()
    return compact, lens


def load_or_create_text_embeddings(
    prompt: str,
    negative_prompt: str,
    model_dir: Path = Path("models/GRN"),
    device_name: str = "mps",
    text_dtype: str = "bf16",
    cache_dtype: str = "fp32",
) -> tuple[mx.array, mx.array]:
    cache_base = _cache_dir(model_dir) / _digest(model_dir, prompt, negative_prompt, text_dtype, cache_dtype)
    cache_key = cache_base.stem
    if cache_key in _TEXT_ARRAY_CACHE:
        return _TEXT_ARRAY_CACHE[cache_key]
    weights_path = cache_base.with_suffix(".safetensors")
    meta_path = cache_base.with_suffix(".json")
    if weights_path.exists() and meta_path.exists():
        arrays = mx.load(str(weights_path))
        result = (arrays["cond"], arrays["uncond"])
        _TEXT_ARRAY_CACHE[cache_key] = result
        return result

    device = torch.device(device_name if device_name != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    text_encoder = _load_t5_encoder(model_dir, device, text_dtype)
    cond, cond_lens = _encode_one(text_encoder, prompt, device, cache_dtype)
    uncond, uncond_lens = _encode_one(text_encoder, negative_prompt, device, cache_dtype)
    text_encoder.model.to("cpu")
    del text_encoder
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    save_file({"cond": cond, "uncond": uncond}, str(weights_path))
    meta_path.write_text(
        json.dumps(
            {
                "cond_lens": cond_lens,
                "uncond_lens": uncond_lens,
                "text_dtype": text_dtype,
                "cache_dtype": cache_dtype,
            },
            indent=2,
        )
    )
    arrays = mx.load(str(weights_path))
    result = (arrays["cond"], arrays["uncond"])
    _TEXT_ARRAY_CACHE[cache_key] = result
    return result


def load_or_create_text_embeddings_with_key(
    prompt: str,
    negative_prompt: str,
    model_dir: Path = Path("models/GRN"),
    device_name: str = "mps",
    text_dtype: str = "bf16",
    cache_dtype: str = "fp32",
) -> tuple[mx.array, mx.array, str]:
    cache_base = _cache_dir(model_dir) / _digest(model_dir, prompt, negative_prompt, text_dtype, cache_dtype)
    cond, uncond = load_or_create_text_embeddings(
        prompt,
        negative_prompt,
        model_dir=model_dir,
        device_name=device_name,
        text_dtype=text_dtype,
        cache_dtype=cache_dtype,
    )
    return cond, uncond, cache_base.stem


def clear_text_embedding_cache() -> None:
    _TEXT_ARRAY_CACHE.clear()
    gc.collect()
