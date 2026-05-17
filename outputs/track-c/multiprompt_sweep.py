"""Multi-prompt K stability sweep for --cfg-start-step.

Runs every (prompt, K) pair with a fixed seed and prompt-specific CLIP labels,
then prints a table. Decision rule: promote K=15 to default only if every
prompt passes positive_score >= 0.93 at K=15.

The script lives under outputs/ because the directory is gitignored. The
findings table is the deliverable, not the script.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from xgrn_mlx.run import generate_mac


# Shared distractors used as CLIP negatives for every prompt.
DISTRACTORS = [
    "a blurry distorted image",
    "an abstract colorful painting",
    "random noise pattern",
    "a black and white photograph",
]


@dataclass
class Prompt:
    name: str
    text: str
    positives: list[str]


PROMPTS: list[Prompt] = [
    Prompt(
        name="orange_tabby",
        text=(
            "A realistic photo of an orange tabby cat sitting on a windowsill, "
            "fluffy fur, green eyes, soft daylight, natural indoor background, "
            "sharp focus, warm but realistic colors"
        ),
        positives=[
            "a realistic photo of an orange tabby cat",
            "a photo of a cat",
        ],
    ),
    Prompt(
        name="red_apple",
        text="A small red apple on a wooden table, natural light",
        positives=[
            "a photo of a red apple on a table",
            "a photo of fruit on a wooden surface",
        ],
    ),
    Prompt(
        name="mountain_lake",
        text=(
            "A serene mountain lake at sunset, golden hour, vibrant colors, "
            "reflections in calm water, distant peaks, natural landscape"
        ),
        positives=[
            "a landscape photo of a mountain lake at sunset",
            "a photo of mountains reflected in water",
        ],
    ),
    Prompt(
        name="bookshelf",
        text=(
            "A cozy bookshelf in a sunlit library, vintage leather books, "
            "warm wooden tones, soft daylight from a window"
        ),
        positives=[
            "a photo of a bookshelf full of books",
            "a photo of a library interior",
        ],
    ),
    Prompt(
        name="elderly_portrait",
        text=(
            "A portrait of an elderly woman with kind eyes and silver hair, "
            "soft natural lighting, neutral background, sharp focus on the face"
        ),
        positives=[
            "a portrait photograph of an elderly woman",
            "a photo of an old person's face",
        ],
    ),
    Prompt(
        name="croissant",
        text=(
            "A freshly baked chocolate croissant on a marble counter, crumbs "
            "scattered, golden flaky pastry, morning light"
        ),
        positives=[
            "a photo of a chocolate croissant",
            "a photo of pastry on a counter",
        ],
    ),
    Prompt(
        name="red_sports_car",
        text=(
            "A red sports car parked on a city street at dusk, glossy paint, "
            "urban background, sharp focus, photographic"
        ),
        positives=[
            "a photo of a red sports car",
            "a photo of a car on a street",
        ],
    ),
]


K_VALUES = [0, 10, 12, 15]
SEED = 42
GATE = 0.93


def clip_scorer(model_id: str = "openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval()

    def score(image_path: Path, positives: list[str]) -> dict:
        labels = positives + DISTRACTORS
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
        scores = {label: float(prob) for label, prob in zip(labels, probs.tolist())}
        positive_score = sum(scores[label] for label in positives)
        negative_score = sum(scores[label] for label in DISTRACTORS)
        top_label = max(scores, key=scores.get)
        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "top_label": top_label,
            "passes_gate": positive_score >= GATE,
        }

    return score


def run_one(prompt: Prompt, k: int, sweep_root: Path) -> dict:
    out_dir = sweep_root / prompt.name / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = generate_mac(
        task="T2I",
        prompt=prompt.text,
        seed=SEED,
        pn="0.25M",
        steps=50,
        guidance=3.0,
        temperature=1.1,
        cfg_start_step=k,
        output_dir=out_dir,
    )
    elapsed = time.perf_counter() - t0
    return {
        "image": str(out_dir / "latest_t2i.png"),
        "end_to_end_sec": elapsed,
        "grn_refine_sec": result.timings.get("grn_refine_sec"),
    }


def main() -> None:
    sweep_root = Path("outputs/track-c/multiprompt_sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    score = clip_scorer()
    start = time.perf_counter()
    # Warm the GRN/decoder caches once with a throwaway run before measurement.
    print("warmup ...", flush=True)
    _ = generate_mac(
        task="T2I",
        prompt=PROMPTS[0].text,
        seed=SEED,
        pn="0.25M",
        steps=2,
        guidance=3.0,
        cfg_start_step=0,
        output_dir=sweep_root / "_warmup",
    )
    for prompt in PROMPTS:
        for k in K_VALUES:
            tag = f"{prompt.name}|K={k}"
            print(f"running {tag} ...", flush=True)
            run = run_one(prompt, k, sweep_root)
            sc = score(Path(run["image"]), prompt.positives)
            row = {
                "prompt": prompt.name,
                "K": k,
                "end_to_end_sec": run["end_to_end_sec"],
                "grn_refine_sec": run["grn_refine_sec"],
                "positive_score": sc["positive_score"],
                "negative_score": sc["negative_score"],
                "top_label": sc["top_label"],
                "passes_gate": sc["passes_gate"],
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
    report_path = sweep_root / "report.json"
    report_path.write_text(json.dumps(
        {
            "seed": SEED,
            "gate": GATE,
            "k_values": K_VALUES,
            "prompts": [p.name for p in PROMPTS],
            "rows": rows,
            "elapsed_sec": time.perf_counter() - start,
        },
        indent=2,
    ))
    print(f"\nwrote {report_path}")
    # Pretty-print pass matrix.
    print("\n==== K stability matrix (positive_score / passes_gate) ====")
    header = f"{'prompt':<18} " + "  ".join(f"K={k:<8}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for prompt in PROMPTS:
        cells = []
        for k in K_VALUES:
            row = next(r for r in rows if r["prompt"] == prompt.name and r["K"] == k)
            mark = "PASS" if row["passes_gate"] else "FAIL"
            cells.append(f"{row['positive_score']:.3f}/{mark:<4}")
        print(f"{prompt.name:<18} " + "  ".join(f"{c:<10}" for c in cells))
    # Decision logic.
    k15_all_pass = all(
        next(r for r in rows if r["prompt"] == p.name and r["K"] == 15)["passes_gate"]
        for p in PROMPTS
    )
    k12_all_pass = all(
        next(r for r in rows if r["prompt"] == p.name and r["K"] == 12)["passes_gate"]
        for p in PROMPTS
    )
    k10_all_pass = all(
        next(r for r in rows if r["prompt"] == p.name and r["K"] == 10)["passes_gate"]
        for p in PROMPTS
    )
    print("\n==== Decision ====")
    print(f"K=10 all-pass: {k10_all_pass}")
    print(f"K=12 all-pass: {k12_all_pass}")
    print(f"K=15 all-pass: {k15_all_pass}")
    if k15_all_pass:
        print("=> Recommend: promote K=15 as default.")
    elif k12_all_pass:
        print("=> Recommend: K=12 as best candidate; K=15 too aggressive for some prompts.")
    elif k10_all_pass:
        print("=> Recommend: K=10 as best candidate; K=15 and K=12 too aggressive for some prompts.")
    else:
        print("=> Recommend: keep K=0 default. No K >= 10 clears every prompt at the 0.93 gate.")


if __name__ == "__main__":
    main()
