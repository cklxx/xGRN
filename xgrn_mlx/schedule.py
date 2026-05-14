from __future__ import annotations

import math

import numpy as np

from .constants import H_DIV_W_TEMPLATES, PN_BASE_SCALE


def nearest_h_div_w(value: float) -> float:
    arr = np.array(H_DIV_W_TEMPLATES)
    return float(arr[np.argmin(np.abs(arr - value))])


def scale_schedule(pn: str, h_div_w: float, num_frames: int) -> tuple[list[tuple[int, int, int]], float]:
    mapped = nearest_h_div_w(h_div_w)
    scale = PN_BASE_SCALE[pn]
    area = scale * scale
    pw = int(round(math.sqrt(area / mapped)))
    ph = int(round(pw * mapped))
    pt = (num_frames - 1) // 4 + 1
    return [(pt, ph, pw)], mapped


def shift_pt(pt: float, alpha: float) -> float:
    if alpha > 1000:
        alpha = alpha - 1000
    noise_pt = 1 - pt
    noise_pt = alpha * noise_pt / (1 + (alpha - 1) * noise_pt)
    return 1 - noise_pt


def refinement_target_pt(step_index: int, steps: int, snr_shift: float = 1.0) -> float:
    pt_unshift = (step_index + 1) / max(1, steps - 1)
    shifted = shift_pt(min(1.0, pt_unshift), snr_shift)
    return float((1 - np.cos(np.pi / 2 * shifted)) * 0.999)
