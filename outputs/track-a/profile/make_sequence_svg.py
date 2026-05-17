"""Hand-rolled sequence diagram SVG of the xGRN generation pipeline.

Actors as vertical lifelines, time flowing top-to-bottom. Activations
(boxes) along each lifeline indicate when that component is busy, with
duration labeled in the box. Messages between lifelines have arrows.
"""

from __future__ import annotations

import json
from pathlib import Path

r0 = json.load(Path("outputs/track-c/sweep/k0/report.json").open())["summary"]
ph = json.load(Path("outputs/track-a/profile/phases.json").open())
bo = json.load(Path("outputs/track-a/profile/block_ops.json").open())

# Numbers (medians, warm)
text_emb_ms = r0["text_embeddings_sec"]["median"] * 1000
model_load_ms = r0["model_load_sec"]["median"] * 1000
decoder_load_ms = r0["decoder_load_sec"]["median"] * 1000
grn_refine_s = r0["grn_refine_sec"]["median"]
hbq_decode_s = r0["hbq_decode_sec"]["median"]
e2e_s = r0["end_to_end_sec"]["median"]

phase = {k: ph[k]["median_ms"] for k in [
    "visual_input_prep", "visual_forward", "logits_head",
    "sampling", "mask_update", "mixed_eval",
]}
step_ms = sum(phase.values())

# Per-op category totals across 28 blocks
ops_per_step = {op: v["total_per_step_ms"] for op, v in bo.items() if op != "_meta"}
mlp_matmul_ms = ops_per_step["mlp_gate_proj"] + ops_per_step["mlp_up_proj"] + ops_per_step["mlp_down_proj"]
qkvo_matmul_ms = sum(ops_per_step[op] for op in ["q_proj", "k_proj", "v_proj", "o_proj"])
sdpa_ms = ops_per_step["sdpa"]
rope_ms = ops_per_step["apply_rope_q"] + ops_per_step["apply_rope_k"]
silu_ms = ops_per_step["mlp_silu_mul"]
norm_ms = sum(ops_per_step[op] for op in ["pre_attn_norm", "post_attn_norm", "q_norm", "k_norm"])
residual_ms = ops_per_step["attn_residual"] + ops_per_step["mlp_residual"]
shape_ms = ops_per_step["qk_transpose"] + ops_per_step["attn_reshape"]
block_total_ms = bo["_meta"]["total_uncompiled_step_ms"]
compiled_step_ms = bo["_meta"]["production_compiled_step_ms"]

# Layout
W = 1500
H = 1000
PAD_T = 90
PAD_B = 70
LIFELINE_TOP = 140
LIFELINE_BOTTOM = H - PAD_B - 20

# Tighter actor spacing on the LEFT; zoom box sits on the RIGHT (x >= 1000).
ACTORS = [
    ("User / xgrn-app",                 110),
    ("Text Encoder\n(UMT5 bf16)",       290),
    ("GRN Model\n(weights + cache)",    470),
    ("visual_forward\n(per step ×28)",  650),
    ("HBQ Decoder\n(native MLX)",       830),
    ("CLIP Validator",                  990),
]
ZOOM_LEFT = 1060  # right column reserved for zoom panel

elems = []

def t(x, y, s, font_size=12, anchor="start", weight="normal", color="#222"):
    # SVG <text> with multi-line via tspan if \n present.
    if "\n" in s:
        lines = s.split("\n")
        parts = [
            f'<tspan x="{x}" dy="{0 if i == 0 else font_size + 2}">{line}</tspan>'
            for i, line in enumerate(lines)
        ]
        body = "".join(parts)
    else:
        body = s
    elems.append(
        f'<text x="{x}" y="{y}" font-family="-apple-system,Helvetica,Arial" '
        f'font-size="{font_size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{color}">{body}</text>'
    )


def rect(x, y, w, h, fill, stroke="#333", stroke_width=0.7, rx=2):
    elems.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def line(x1, y1, x2, y2, color="#888", dash=False, width=1, arrow=False, arrow_color=None):
    da = ' stroke-dasharray="4,3"' if dash else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    elems.append(
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{da}{marker}/>'
    )


# Begin SVG
elems.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
elems.append(
    '<defs>'
    '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
    'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#444"/></marker>'
    '</defs>'
)
rect(0, 0, W, H, "#fafafa", stroke="none")

# Title block
t(W // 2, 36, "xGRN generation — sequence diagram (warm t2i-correct: 0.25M, 50 steps, bf16)",
  font_size=20, anchor="middle", weight="bold")
t(W // 2, 58, f"end-to-end = {e2e_s:.2f} s   |   grn_refine = {grn_refine_s:.2f} s ({grn_refine_s/e2e_s*100:.1f}%)   "
              f"|   hbq_decode = {hbq_decode_s:.2f} s   |   per-step ≈ {step_ms:.0f} ms",
  font_size=12, anchor="middle", color="#555")

# Actor headers + lifelines
for name, x in ACTORS:
    # Header pill
    pill_w = 170
    rect(x - pill_w // 2, 84, pill_w, 36, "#ffffff", stroke="#222", stroke_width=1.2)
    t(x, 100, name, font_size=11, anchor="middle", weight="bold")
    # Lifeline
    line(x, 120, x, LIFELINE_BOTTOM, "#999", dash=True, width=1)


def y_at(frac):
    return LIFELINE_TOP + frac * (LIFELINE_BOTTOM - LIFELINE_TOP)


def msg(from_x, to_x, y, label, label_above=True, color="#444", note=None, note_side="right"):
    """Arrow from from_x to to_x at vertical y, with label."""
    line(from_x, y, to_x, y, color, arrow=True, width=1.5)
    label_y = y - 8 if label_above else y + 18
    label_x = (from_x + to_x) / 2
    t(label_x, label_y, label, font_size=11, anchor="middle", color="#222")
    if note:
        nx = to_x + 14 if note_side == "right" else from_x - 14
        anchor = "start" if note_side == "right" else "end"
        t(nx, y + 4, note, font_size=10, color="#666", anchor=anchor)


def activation(x, y_top, y_bottom, fill="#eaf2fb", stroke="#2a66ad"):
    """Vertical activation box on a lifeline."""
    rect(x - 6, y_top, 12, y_bottom - y_top, fill, stroke=stroke, stroke_width=0.8, rx=2)


# --- Sequence content -------------------------------------------------

USER, TEXT, MODEL, VIS, DEC, CLIP = (a[1] for a in ACTORS)

# Step counters: 9 steps from 140 to LIFELINE_BOTTOM-30
step_count = 10
def yi(i):
    return LIFELINE_TOP + i * (LIFELINE_BOTTOM - LIFELINE_TOP - 30) / step_count


# 0: prompt
msg(USER, TEXT, yi(0), "prompt + negative", note=f"text_emb {text_emb_ms:.2f} ms (warm cache)")

# 1: encoded
msg(TEXT, USER, yi(1), "cond / uncond embeds")

# 2: load model
msg(USER, MODEL, yi(2), "load weights + cfg_kv_cache", note=f"model_load {model_load_ms:.2f} ms")

# 3: load decoder
msg(USER, DEC, yi(3), "load HBQ", note=f"decoder_load {decoder_load_ms:.2f} ms")

# 4: enter refinement loop
loop_top = yi(4) - 10
loop_bot = yi(8) + 10
rect(USER - 30, loop_top, VIS - USER + 150, loop_bot - loop_top, "#fff8e7",
     stroke="#d29c2a", stroke_width=1.0)
t(USER - 22, loop_top + 14, "loop  ×50 steps  ≈ 75.6 s  (per step ≈ 1.51 s)",
  font_size=11, weight="bold", color="#7a5500")

# 5: visual_input_prep
msg(USER, VIS, yi(5), "visual_input  [B, L, H]",
    note=f"prep {phase['visual_input_prep']:.2f} ms")

# 6: 28-block forward
msg(VIS, MODEL, yi(6), "28 × block(x, rope, kv, mask)",
    note=f"visual_forward 1510 ms  ★ 99.5%",
    note_side="right")
# Activation box on the visual_forward lifeline
activation(VIS, yi(6) + 8, yi(7) - 8, fill="#1f4e8b", stroke="#163b6a")

# 7: cfg_out
msg(MODEL, VIS, yi(7) - 6, "cfg_out  →  logits → categorical → mask",
    note=f"logits+sample+mask ≈ {phase['logits_head'] + phase['sampling'] + phase['mask_update']:.2f} ms")

# 8: pred labels
msg(VIS, USER, yi(8), "pred_labels / mixed")

# 9 (outside loop): decode
msg(USER, DEC, yi(9), "bit_labels → raw frame", note=f"hbq_decode {hbq_decode_s*1000:.0f} ms")

# 10 final: decoded image
msg(DEC, USER, yi(9) + 28, "image PNG  +  refinement_stats.csv")

# 11 validate
msg(USER, CLIP, yi(9) + 56, "validate(latest_t2i.png)", note="CLIP positive 0.9904  ✓")


# --- Per-step zoom box on the right column ------------------------

zoom_x0 = ZOOM_LEFT
zoom_y0 = LIFELINE_TOP + 20
zoom_w = W - ZOOM_LEFT - 30
zoom_h = 360
rect(zoom_x0, zoom_y0, zoom_w, zoom_h, "#ffffff", stroke="#1f4e8b", stroke_width=1.5)
t(zoom_x0 + zoom_w // 2, zoom_y0 + 20,
  "visual_forward op breakdown (per step, summed over 28 blocks)",
  font_size=12, anchor="middle", weight="bold")
t(zoom_x0 + zoom_w // 2, zoom_y0 + 36,
  f"uncompiled total {block_total_ms:.0f} ms  ·  compiled {compiled_step_ms:.0f} ms",
  font_size=10, anchor="middle", color="#666")

categories = [
    ("MLP matmuls (gate+up+down)", mlp_matmul_ms, "#1f4e8b"),
    ("QKV/O matmuls",              qkvo_matmul_ms, "#5b9bd5"),
    ("SDPA",                       sdpa_ms,        "#8e4ec6"),
    ("apply_rope (q+k)",           rope_ms,        "#e07b3c"),
    ("silu·up",                    silu_ms,        "#3d8884"),
    ("norms (rms_norm)",           norm_ms,        "#5cb85c"),
    ("residual + shape ops",       residual_ms + shape_ms, "#bdbdbd"),
]
cat_total = sum(v for _, v, _ in categories)
row_y = zoom_y0 + 56
for label, val, color in categories:
    pct = val / cat_total * 100
    bar_max_w = 260
    bar_w = val / cat_total * bar_max_w
    rect(zoom_x0 + 14, row_y - 9, bar_w, 14, color)
    t(zoom_x0 + 14 + bar_max_w + 8, row_y, f"{val:5.0f} ms  ({pct:.0f}%)",
      font_size=10, color="#222")
    t(zoom_x0 + 14, row_y - 12, label, font_size=10, color="#333")
    row_y += 32


# Footer
t(40, H - 20,
  "Headline:  99.5 % of step is visual_forward.  78 % of visual_forward is matmuls "
  "(MLP alone 56 %).  Track A1-lite fused apply_rope shipped (-1.63 % wall, opt-in).  "
  "All QKV / MLP fusion attempts regressed at the strict gate.",
  font_size=11, color="#444")
t(40, H - 6,
  "Source: outputs/track-c/sweep/k0 + outputs/track-a/profile/{phases,block_ops}.json   ·   "
  "Sequence diagram hand-composed; bar/zoom is data-driven.",
  font_size=10, color="#999")

elems.append("</svg>")

OUT = Path("outputs/track-a/profile/pipeline_sequence.svg")
OUT.write_text("\n".join(elems))
print(f"Wrote {OUT}  ({OUT.stat().st_size} bytes)")
