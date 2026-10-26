"""Run one inference against an image file and print the model result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from gradcam import predict, load_model_once


def main() -> None:
    """Load the model and report its prediction for the requested image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a JPG or PNG image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parents[1] / "model" / "leaf_classifier_fp32.keras",
        help="Path to leaf_classifier_fp32.keras",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    model = load_model_once(args.model)
    image_rgb = np.asarray(Image.open(args.image).convert("RGB"))
    result = predict(image_rgb, model)
    print(f"label: {result['label']}")
    print(f"confidence: {result['confidence']:.4f}")
    print("probabilities:")
    for label, probability in result["probabilities"].items():
        print(f"  {label}: {probability:.4f}")
    print(f"box: {result['box']}")
    print(f"status: {result['status']}")
    print(f"inference_ms: {result['inference_ms']:.2f}")


if __name__ == "__main__":
    main()
