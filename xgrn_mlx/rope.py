from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

from .constants import GRN2B


@lru_cache(maxsize=1)
def rope_tables() -> dict[str, mx.array]:
    cfg = GRN2B
    dim = cfg.head_dim
    dim_div_2 = dim // 2
    num_former = dim_div_2 // 3
    num_last = dim_div_2 - num_former * 2
    inv_former = 1.0 / (cfg.rope_base ** (np.arange(num_former, dtype=np.float32) / num_former))
    inv_last = 1.0 / (cfg.rope_base ** (np.arange(num_last, dtype=np.float32) / num_last))
    t_frames = np.arange(cfg.rope_text_len + cfg.rope_max_frames, dtype=np.float32)
    t_height = np.arange(cfg.rope_max_height, dtype=np.float32)
    t_width = np.arange(cfg.rope_max_width, dtype=np.float32)

    frames = np.outer(t_frames, inv_former)
    height = np.outer(t_height, inv_former)
    width = np.outer(t_width, inv_last)
    freqs_frames = np.stack([np.cos(frames), np.sin(frames)], axis=0).astype(np.float16)
    freqs_height = np.stack([np.cos(height), np.sin(height)], axis=0).astype(np.float16)
    freqs_width = np.stack([np.cos(width), np.sin(width)], axis=0).astype(np.float16)

    tm = cfg.rope_text_len
    text = np.concatenate(
        [
            np.broadcast_to(freqs_frames[:, :tm, None, None, :], (2, tm, 1, 1, num_former)),
            np.broadcast_to(freqs_height[:, None, :1, None, :], (2, tm, 1, 1, num_former)),
            np.broadcast_to(freqs_width[:, None, None, :1, :], (2, tm, 1, 1, num_last)),
        ],
        axis=-1,
    ).reshape(2, tm, dim_div_2)
    return {
        "text": mx.array(text),
        "frames": mx.array(freqs_frames[:, tm:]),
        "height": mx.array(freqs_height),
        "width": mx.array(freqs_width),
    }


def text_rope(length: int, offset: int = 0) -> mx.array:
    return rope_tables()["text"][:, offset : offset + length]


def visual_rope(pt: int, ph: int, pw: int, mapped_h_div_w: float) -> mx.array:
    tables = rope_tables()
    max_height = GRN2B.rope_max_height
    extreme_h_div_w = 3.0
    extreme_h = max_height
    extreme_w = extreme_h / extreme_h_div_w
    upw = int(math.sqrt(extreme_h * extreme_w / mapped_h_div_w))
    uph = int(mapped_h_div_w * upw)
    frame = tables["frames"][:, :pt]
    h_idx = np.rint(np.arange(ph) * (uph / ph)).astype(np.int32)
    w_idx = np.rint(np.arange(pw) * (upw / pw)).astype(np.int32)
    height = mx.take(tables["height"], mx.array(h_idx), axis=1)
    width = mx.take(tables["width"], mx.array(w_idx), axis=1)
    f = mx.broadcast_to(frame[:, :, None, None, :], (2, pt, ph, pw, frame.shape[-1]))
    h = mx.broadcast_to(height[:, None, :, None, :], (2, pt, ph, pw, height.shape[-1]))
    w = mx.broadcast_to(width[:, None, None, :, :], (2, pt, ph, pw, width.shape[-1]))
    return mx.concatenate([f, h, w], axis=-1).reshape(2, pt * ph * pw, GRN2B.head_dim // 2)


def apply_rope(x: mx.array, rope: mx.array) -> mx.array:
    # x: [B, H, L, D], rope: [2, L, D/2]
    half = x.shape[-1] // 2
    xr = x.reshape(*x.shape[:-1], half, 2)
    cos = rope[0].reshape(1, 1, rope.shape[1], half)
    sin = rope[1].reshape(1, 1, rope.shape[1], half)
    a = xr[..., 0] * cos - xr[..., 1] * sin
    b = xr[..., 0] * sin + xr[..., 1] * cos
    return mx.stack([a, b], axis=-1).reshape(x.shape)
