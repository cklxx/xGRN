from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from PIL import Image


REFERENCE_GRN = (Path(__file__).resolve().parents[1] / "../GRN").resolve()


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


def _vae_args() -> SimpleNamespace:
    return SimpleNamespace(detail_scale_dim=64)


class HBQMPSDecoder:
    def __init__(self, model_dir: Path = Path("models/GRN"), device_name: str = "mps"):
        _ensure_reference_importable()
        from grn.models.hbq_tokenizer import HBQ_Tokenizer

        self.device = torch.device(device_name if device_name != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.vae = HBQ_Tokenizer(args=_vae_args(), latent_channels=64, encoder_out_type="feature_tanh")
        state = torch.load(model_dir / "HBQ_tokenizer_64dim_M4.ckpt", map_location="cpu", mmap=True, weights_only=False)
        state = state["ema"] if "ema" in state else state["vae"]
        self.vae.load_state_dict(state, assign=True)
        self.vae.eval().requires_grad_(False)
        self.vae.to(device=self.device, dtype=self.dtype)
        del state
        gc.collect()

    def decode_tensor(self, raw: mx.array) -> torch.Tensor:
        arr = np.array(raw.astype(mx.float16))
        tensor = torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            image = self.vae.decode(tensor, slice=True)
        image = torch.clamp((image + 1) / 2, 0, 1)
        return image.detach().cpu()

    def decode_first_image(self, raw: mx.array) -> Image.Image:
        tensor = self.decode_tensor(raw)
        frame = tensor[0, :, 0].permute(1, 2, 0).numpy()
        return Image.fromarray(np.round(frame * 255).clip(0, 255).astype(np.uint8), mode="RGB")

    def close(self) -> None:
        self.vae.to("cpu")
        del self.vae
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()

