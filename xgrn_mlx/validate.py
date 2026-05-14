from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


POSITIVE_LABELS = [
    "a realistic photo of an orange tabby cat",
    "a photo of a cat",
]

NEGATIVE_LABELS = [
    "a blurry distorted image",
    "an abstract colorful painting",
    "a landscape photo",
    "a breakfast table with food",
]


@lru_cache(maxsize=4)
def _load_clip(model_id: str) -> tuple:
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval()
    return processor, model


def validate_image(
    image_path: Path,
    *,
    model_id: str = "openai/clip-vit-base-patch32",
    min_positive: float = 0.5,
) -> dict:
    labels = POSITIVE_LABELS + NEGATIVE_LABELS
    processor, model = _load_clip(model_id)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    scores = {label: float(prob) for label, prob in zip(labels, probs.tolist())}
    positive_score = sum(scores[label] for label in POSITIVE_LABELS)
    negative_score = sum(scores[label] for label in NEGATIVE_LABELS)
    top_label = max(scores, key=scores.get)
    passed = top_label in POSITIVE_LABELS and positive_score >= min_positive and positive_score > negative_score
    return {
        "image": str(image_path),
        "passed": passed,
        "top_label": top_label,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate xGRN orange-cat image semantics with CLIP.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("outputs/latest_t2i.png"))
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--min-positive", type=float, default=0.5)
    args = parser.parse_args()
    result = validate_image(args.image, model_id=args.model_id, min_positive=args.min_positive)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
