"""Verify the supplied leaf classifier loads and produces four probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    """Load the model, inspect its contract, and run one random prediction."""
    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "model" / "leaf_classifier_fp32.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print("Layer names:")
    for layer in model.layers:
        print(f"  {layer.name}")

    random_image = np.random.default_rng().uniform(
        0.0, 255.0, size=(1, 384, 384, 3)
    ).astype(np.float32)
    probabilities = np.asarray(model.predict(random_image, verbose=0))
    if probabilities.shape != (1, 4):
        raise AssertionError(
            f"Expected prediction shape (1, 4), got {probabilities.shape}"
        )
    row = probabilities[0]
    probability_sum = float(np.sum(row))
    print(f"Output probabilities: {row.tolist()}")
    print(f"Probability sum: {probability_sum:.6f}")
    if not np.allclose(probability_sum, 1.0, atol=1e-3):
        raise AssertionError(
            f"Expected probabilities to sum to about 1, got {probability_sum}"
        )
    print("MODEL OK")


if __name__ == "__main__":
    main()
