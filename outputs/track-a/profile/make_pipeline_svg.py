"""Compose a 3-level drill-down SVG of the xGRN pipeline timing.

Reads:
  - outputs/track-c/sweep/k0/report.json     (whole pipeline warm timings)
  - outputs/track-a/profile/phases.json      (per-step phase breakdown)
  - outputs/track-a/profile/block_ops.json   (per-op-per-block breakdown)

Emits:
  outputs/track-a/profile/pipeline_timing.svg

Layout:
  L0  end-to-end (76.66 s)              -> grn_refine highlighted
  L1  one refinement step (1.51 s)      -> visual_forward highlighted
  L2  one block summed over 28          -> per-op breakdown
  Tiny segments use leader lines instead of overlapping inline labels.
"""

from __future__ import annotations

import json
from pathlib import Path


# --- Read profile data --------------------------------------------------

ROOT = Path("outputs/track-c/sweep/k0/report.json")
PHASES = Path("outputs/track-a/profile/phases.json")
BLOCK_OPS = Path("outputs/track-a/profile/block_ops.json")

r0 = json.load(ROOT.open())["summary"]
ph = json.load(PHASES.open())
bo = json.load(BLOCK_OPS.open())

e2e_total = r0["end_to_end_sec"]["median"]
grn_refine = r0["grn_refine_sec"]["median"]
hbq_decode = r0["hbq_decode_sec"]["median"]
text_emb = r0["text_embeddings_sec"]["median"]
model_load = r0["model_load_sec"]["median"]
decoder_load = r0["decoder_load_sec"]["median"]
image_mat = r0["image_materialize_sec"]["median"]
output_write = r0["output_write_sec"]["median"]
clip_validation_sec = max(
    0.0,
    e2e_total - grn_refine - hbq_decode - text_emb - model_load - decoder_load
    - image_mat - output_write,
)

phase_keys = [
    "visual_input_prep", "visual_forward", "logits_head",
    "sampling", "mask_update", "mixed_eval",
]
step_phases_ms = {k: ph[k]["median_ms"] for k in phase_keys}
step_total_ms = sum(step_phases_ms.values())

ops = [(op, v["total_per_step_ms"]) for op, v in bo.items() if op != "_meta"]
ops.sort(key=lambda x: -x[1])
uncompiled_step_ms = bo["_meta"]["total_uncompiled_step_ms"]
compiled_step_ms = bo["_meta"]["production_compiled_step_ms"]


# --- Color scheme ------------------------------------------------------

OP_COLORS = {
    "mlp_down_proj":   "#1f4e8b",
    "mlp_up_proj":     "#2a66ad",
    "mlp_gate_proj":   "#367cca",
    "q_proj":          "#5b9bd5",
    "k_proj":          "#5b9bd5",
    "v_proj":          "#5b9bd5",
    "o_proj":          "#5b9bd5",
    "sdpa":            "#8e4ec6",
    "mlp_silu_mul":    "#3d8884",
    "pre_attn_norm":   "#5cb85c",
    "post_attn_norm":  "#5cb85c",
    "q_norm":          "#80c780",
    "k_norm":          "#80c780",
    "apply_rope_q":    "#e07b3c",
    "apply_rope_k":    "#e07b3c",
    "attn_residual":   "#bdbdbd",
    "mlp_residual":    "#bdbdbd",
    "qk_transpose":    "#d9d9d9",
    "attn_reshape":    "#d9d9d9",
}

PHASE_COLORS = {
    "visual_input_prep": "#dcdcdc",
    "visual_forward":    "#1f4e8b",
    "logits_head":       "#5b9bd5",
    "sampling":          "#f1c40f",
    "mask_update":       "#f39c12",
    "mixed_eval":        "#999999",
}


# --- Canvas + layout ---------------------------------------------------

W = 1280
PAD_LEFT = 40
PAD_RIGHT = 40
BAR_W = W - PAD_LEFT - PAD_RIGHT

TITLE_H = 70
ROW_H = 270
LEGEND_H = 80
FOOTER_H = 28
H = TITLE_H + 3 * ROW_H + LEGEND_H + FOOTER_H

BAR_H = 50


# --- Drawing helpers ---------------------------------------------------

elems = []

def t_(x, y, s, font_size=12, anchor="start", weight="normal", color="#222"):
    elems.append(
        f'<text x="{x}" y="{y}" font-family="-apple-system,Helvetica,Arial" '
        f'font-size="{font_size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{color}">{s}</text>'
    )


def r_(x, y, w, h, fill, stroke="#333", stroke_width=0.5):
    elems.append(
        f'<rect x="{x:.2f}" y="{y}" width="{max(w, 0):.2f}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def l_(x1, y1, x2, y2, color="#888", dash=False, width=1):
    da = ' stroke-dasharray="3,3"' if dash else ""
    elems.append(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{da}/>'
    )


def fmt_seconds(s):
    if s >= 1.0:
        return f"{s:.2f} s"
    if s >= 0.001:
        return f"{s * 1000:.1f} ms"
    if s > 0:
        return f"{s * 1e6:.0f} µs"
    return "0"


def fmt_ms(ms):
    if ms >= 100.0:
        return f"{ms:.0f} ms"
    if ms >= 10.0:
        return f"{ms:.1f} ms"
    if ms >= 0.1:
        return f"{ms:.2f} ms"
    return f"{ms:.3f} ms"


def axis(y, total, fmt, n_ticks=5):
    """Bottom axis with ticks."""
    l_(PAD_LEFT, y, PAD_LEFT + BAR_W, y, "#bbb")
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        x = PAD_LEFT + frac * BAR_W
        l_(x, y - 4, x, y + 4, "#bbb")
        t_(x, y + 16, fmt(total * frac), font_size=10, anchor="middle", color="#777")


def draw_row(y0, title, subtitle, segments, total, fmt, inline_min_px=42,
             use_leaders_below=True):
    """One drill-down row. segments = [(name, value, color)]."""
    t_(PAD_LEFT, y0 + 22, title, font_size=18, weight="bold")
    t_(PAD_LEFT, y0 + 44, subtitle, font_size=11, color="#666")
    bar_top = y0 + 66

    # Draw bars (skip invisible ones).
    x = PAD_LEFT
    bars = []
    for name, val, color in segments:
        w = val / total * BAR_W
        if w < 0.4:
            x += w
            continue
        r_(x, bar_top, w, BAR_H, color)
        bars.append((name, val, x, w, color))
        x += w

    # Inline labels for bars wide enough.
    for name, val, bx, bw, color in bars:
        center = bx + bw / 2
        if bw >= inline_min_px:
            light_bg = color in ("#dcdcdc", "#d9d9d9", "#bdbdbd", "#999999",
                                 "#5b9bd5", "#80c780", "#f1c40f", "#f39c12",
                                 "#5cb85c", "#3d8884", "#e07b3c")
            label_color = "#222" if light_bg else "#fff"
            t_(center, bar_top + BAR_H / 2 + 4, fmt(val),
               font_size=11, anchor="middle", color=label_color, weight="bold")
            t_(center, bar_top - 6, name, font_size=10, anchor="middle", color="#333")

    # Leader lines + offset labels for too-small bars (per-op breakdown row).
    if use_leaders_below:
        small_bars = [b for b in bars if b[3] < inline_min_px]
        if small_bars:
            label_y = bar_top + BAR_H + 38
            # spread offsets evenly across the small region
            for i, (name, val, bx, bw, color) in enumerate(small_bars):
                center = bx + bw / 2
                # zig-zag staggering to avoid overlap
                step = i
                ly = label_y + (step % 3) * 14
                # leader line
                l_(center, bar_top + BAR_H, center, ly - 11, "#aaa", dash=False, width=0.5)
                t_(center, ly, f"{name} {fmt(val)}",
                   font_size=9, anchor="middle", color="#444")


# --- Compose -----------------------------------------------------------

elems.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">'
)
r_(0, 0, W, H, "#fafafa", stroke="none")

t_(W // 2, 32,
   "xGRN — generation pipeline timing (warm t2i-correct: 0.25M, 50 steps, bf16)",
   font_size=20, anchor="middle", weight="bold")
t_(W // 2, 54,
   "From outputs/track-c/sweep/k0 + outputs/track-a/profile/{phases,block_ops}.json",
   font_size=11, anchor="middle", color="#666")


# Row 0 — end-to-end
L0_SEGMENTS = [
    ("setup (warm-cached, ~0)", text_emb + model_load + decoder_load, "#dcdcdc"),
    ("grn_refine (50 steps)",   grn_refine,                            "#1f4e8b"),
    ("hbq_decode",              hbq_decode,                            "#8e4ec6"),
    ("write + clip validate",   image_mat + output_write + clip_validation_sec,
                                                                       "#5cb85c"),
]
y0 = TITLE_H
draw_row(
    y0=y0,
    title="① End-to-end (one warm generation)",
    subtitle=f"total = {fmt_seconds(e2e_total)}  ·  grn_refine = {grn_refine/e2e_total*100:.1f}% of pipeline",
    segments=L0_SEGMENTS,
    total=e2e_total,
    fmt=fmt_seconds,
    inline_min_px=40,
    use_leaders_below=False,
)
# Axis under L0
axis(y0 + 66 + BAR_H + 10, e2e_total, fmt_seconds)

# Annotate the tiny hbq_decode segment with a leader
setup_s = text_emb + model_load + decoder_load
grn_x_end = PAD_LEFT + (setup_s + grn_refine) / e2e_total * BAR_W
hbq_x_center = grn_x_end + (hbq_decode / e2e_total * BAR_W) / 2
l_(hbq_x_center, y0 + 66 + BAR_H, hbq_x_center - 30, y0 + 66 + BAR_H + 50, "#aaa", width=0.5)
t_(hbq_x_center - 32, y0 + 66 + BAR_H + 62,
   f"hbq_decode {fmt_seconds(hbq_decode)}",
   font_size=9, anchor="middle", color="#444")
write_x_center = grn_x_end + (hbq_decode / e2e_total * BAR_W) + ((image_mat + output_write + clip_validation_sec) / e2e_total * BAR_W) / 2
l_(write_x_center, y0 + 66 + BAR_H, write_x_center, y0 + 66 + BAR_H + 76, "#aaa", width=0.5)
t_(write_x_center, y0 + 66 + BAR_H + 88,
   f"write+clip {fmt_seconds(image_mat + output_write + clip_validation_sec)}",
   font_size=9, anchor="middle", color="#444")

# Bracket between L0 grn_refine and L1
l0_y_bottom = y0 + 66 + BAR_H
l1_y_top = y0 + ROW_H + 66
grn_x_start = PAD_LEFT + setup_s / e2e_total * BAR_W
l_(grn_x_start, l0_y_bottom + 100, PAD_LEFT, l1_y_top, "#888", dash=True)
l_(grn_x_end,   l0_y_bottom + 100, PAD_LEFT + BAR_W, l1_y_top, "#888", dash=True)


# Row 1 — one refinement step
L1_SEGMENTS = [(k, step_phases_ms[k], PHASE_COLORS[k]) for k in phase_keys]
y1 = TITLE_H + ROW_H
draw_row(
    y0=y1,
    title="② One refinement step (1 of 50)",
    subtitle=f"step = {fmt_ms(step_total_ms)}  ·  visual_forward = {step_phases_ms['visual_forward']/step_total_ms*100:.1f}% of step",
    segments=L1_SEGMENTS,
    total=step_total_ms,
    fmt=fmt_ms,
    inline_min_px=40,
    use_leaders_below=True,
)
axis(y1 + 66 + BAR_H + 10, step_total_ms, fmt_ms)

# Bracket between L1 visual_forward and L2
vis_x_start = PAD_LEFT + step_phases_ms["visual_input_prep"] / step_total_ms * BAR_W
vis_x_end = vis_x_start + step_phases_ms["visual_forward"] / step_total_ms * BAR_W
l1_y_bottom = y1 + 66 + BAR_H
l2_y_top = TITLE_H + 2 * ROW_H + 66
l_(vis_x_start, l1_y_bottom + 100, PAD_LEFT, l2_y_top, "#888", dash=True)
l_(vis_x_end,   l1_y_bottom + 100, PAD_LEFT + BAR_W, l2_y_top, "#888", dash=True)


# Row 2 — per-op breakdown across 28 blocks
L2_SEGMENTS = [(op, val, OP_COLORS.get(op, "#888")) for op, val in ops]
y2 = TITLE_H + 2 * ROW_H
draw_row(
    y0=y2,
    title="③ Per-op breakdown across all 28 blocks (uncompiled visual_forward)",
    subtitle=(f"uncompiled total = {fmt_ms(uncompiled_step_ms)}  ·  "
              f"production compiled = {fmt_ms(compiled_step_ms)}  ·  "
              f"compile saves {(uncompiled_step_ms - compiled_step_ms)/uncompiled_step_ms*100:.1f}%"),
    segments=L2_SEGMENTS,
    total=uncompiled_step_ms,
    fmt=fmt_ms,
    inline_min_px=46,
    use_leaders_below=True,
)
axis(y2 + 66 + BAR_H + 10, uncompiled_step_ms, fmt_ms)


# --- Legend (wraps to multiple rows if needed) -------------------------

legend_y = TITLE_H + 3 * ROW_H + 16
t_(PAD_LEFT, legend_y, "Color groups:", font_size=12, weight="bold")
legend = [
    ("MLP matmuls (gate / up / down)", "#1f4e8b"),
    ("attention QKV / O matmuls", "#5b9bd5"),
    ("SDPA (fused attention)", "#8e4ec6"),
    ("MLP silu·up (elementwise)", "#3d8884"),
    ("norms (rms_norm)", "#5cb85c"),
    ("RoPE (q/k rotation)", "#e07b3c"),
    ("residual / transpose / reshape", "#bdbdbd"),
    ("sampling / mask update", "#f1c40f"),
]
lx = PAD_LEFT + 96
ly = legend_y
for label, color in legend:
    swatch_w = 12
    text_advance = 7 * len(label) + 8
    if lx + swatch_w + text_advance > W - PAD_RIGHT:
        ly += 22
        lx = PAD_LEFT + 96
    r_(lx, ly - 11, swatch_w, 12, color, stroke="#888", stroke_width=0.5)
    t_(lx + swatch_w + 6, ly, label, font_size=11, color="#333")
    lx += swatch_w + text_advance + 16

# Footer line
foot_y = TITLE_H + 3 * ROW_H + LEGEND_H + 18
t_(PAD_LEFT, foot_y,
   "Headline: 99.5% of step is visual_forward.  78% of visual_forward is matmuls.  "
   "MLP alone is 56%.  A1-lite fused apply_rope shipped (-1.63% wall, opt-in); "
   "every QKV / MLP fusion attempt regressed at the strict gate.",
   font_size=11, color="#555")

elems.append("</svg>")

OUT = Path("outputs/track-a/profile/pipeline_timing.svg")
OUT.write_text("\n".join(elems))
print(f"Wrote {OUT}  ({OUT.stat().st_size} bytes, {H}px tall)")
