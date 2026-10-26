"""Model loading, prediction, and Grad-CAM localization for apple leaves."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES: tuple[str, ...] = ("healthy", "multiple_diseases", "rust", "scab")
INPUT_SIZE: tuple[int, int] = (384, 384)
CONFIDENCE_THRESHOLD: float = 0.60
CAM_THRESHOLD: float = 0.50
GREEN_FRACTION_THRESHOLD: float = 0.08
CAM_PADDING_FRACTION: float = 0.05
GREEN_HSV_LOWER: tuple[int, int, int] = (35, 40, 40)
GREEN_HSV_UPPER: tuple[int, int, int] = (85, 255, 255)
MORPH_KERNEL_SIZE: int = 5
DEFAULT_MODEL_PATH = (
    Path(__file__).parent / "model" / "leaf_classifier_fp32.keras"
)

_MODEL_CACHE: dict[str, tf.keras.Model] = {}
_RUNNERS: dict[int, tuple[tf.keras.Model, "_GradCamRunner"]] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_DEFAULT_MODEL: tf.keras.Model | None = None


class _GradCamRunner:
    """Run the backbone and a logits head to produce probabilities and Grad-CAM."""

    def __init__(self, model: tf.keras.Model) -> None:
        if len(model.layers) < 4:
            raise ValueError("The model does not have the expected classifier layers.")

        self.backbone = model.layers[1]
        self.pool = model.layers[2]
        classifier = model.layers[-1]
        if not isinstance(classifier, tf.keras.layers.Dense):
            raise ValueError("The final model layer must be a Dense classifier.")

        self.logit_dense = tf.keras.layers.Dense(
            units=classifier.units,
            activation=None,
            use_bias=classifier.use_bias,
            name="gradcam_logits",
        )
        self.logit_dense.build((None, int(classifier.kernel.shape[0])))
        self.logit_dense.set_weights(classifier.get_weights())
        self._compiled_forward = tf.function(self._forward, reduce_retracing=True)

    def _forward(self, batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Return softmax probabilities and a normalized Grad-CAM map."""
        with tf.GradientTape() as tape:
            feature_map = self.backbone(batch, training=False)
            tape.watch(feature_map)
            pooled = self.pool(feature_map)
            logits = self.logit_dense(pooled)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predicted_classes = tf.argmax(
                probabilities, axis=-1, output_type=tf.int32
            )
            predicted_logits = tf.gather(
                logits, predicted_classes, axis=1, batch_dims=1
            )
            objective = tf.reduce_sum(predicted_logits)

        gradients = tape.gradient(objective, feature_map)
        channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(
            tf.reduce_sum(channel_weights * feature_map, axis=-1)
        )
        minimum = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
        maximum = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
        cam = tf.math.divide_no_nan(cam - minimum, maximum - minimum)
        return probabilities, cam

    def run(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one batch and return NumPy probabilities and CAMs."""
        probabilities, cams = self._compiled_forward(batch)
        return probabilities.numpy(), cams.numpy()


def _validate_model(model: tf.keras.Model) -> None:
    """Reject models that do not match the documented input/output contract."""
    input_shape = model.input_shape
    output_shape = model.output_shape
    if (
        len(input_shape) != 4
        or tuple(input_shape[1:]) != (384, 384, 3)
        or len(output_shape) != 2
        or output_shape[-1] != len(CLASS_NAMES)
    ):
        raise ValueError(
            "Expected model input (None, 384, 384, 3) and four class outputs; "
            f"got input {input_shape} and output {output_shape}."
        )


def load_model_once(path: str | Path) -> tf.keras.Model:
    """Load, validate, and warm the Keras model once for each resolved path."""
    global _DEFAULT_MODEL

    resolved_path = str(Path(path).expanduser().resolve())
    with _MODEL_CACHE_LOCK:
        if resolved_path in _MODEL_CACHE:
            model = _MODEL_CACHE[resolved_path]
            _DEFAULT_MODEL = model
            return model
        if not Path(resolved_path).is_file():
            raise FileNotFoundError(f"Model file not found: {resolved_path}")

        model = tf.keras.models.load_model(resolved_path, compile=False)
        _validate_model(model)
        # Warm the supplied model and compiled Grad-CAM graph with raw 0-255 input.
        model(np.zeros((1, *INPUT_SIZE, 3), dtype=np.float32), training=False)
        runner = _GradCamRunner(model)
        runner.run(np.zeros((1, *INPUT_SIZE, 3), dtype=np.float32))
        _MODEL_CACHE[resolved_path] = model
        _RUNNERS[id(model)] = (model, runner)
        _DEFAULT_MODEL = model
        return model


def _get_runner(model: tf.keras.Model) -> _GradCamRunner:
    """Return the compiled inference runner associated with a loaded model."""
    with _MODEL_CACHE_LOCK:
        cached = _RUNNERS.get(id(model))
        if cached is None or cached[0] is not model:
            _validate_model(model)
            runner = _GradCamRunner(model)
            _RUNNERS[id(model)] = (model, runner)
            return runner
        return cached[1]


def _green_fraction(frame_rgb: np.ndarray) -> float:
    """Measure the fraction of pixels in a broad green HSV range."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(GREEN_HSV_LOWER, dtype=np.uint8),
        np.array(GREEN_HSV_UPPER, dtype=np.uint8),
    )
    return float(np.count_nonzero(mask)) / float(mask.size)


def _extract_box(cam: np.ndarray, threshold: float) -> tuple[int, int, int, int] | None:
    """Extract a padded box around the largest thresholded CAM component."""
    mask = np.asarray(cam >= threshold, dtype=np.uint8) * 255
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if component_count <= 1:
        return None

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    if area <= 0:
        return None

    image_height, image_width = mask.shape
    pad_x = max(1, int(round(width * CAM_PADDING_FRACTION)))
    pad_y = max(1, int(round(height * CAM_PADDING_FRACTION)))
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(image_width - 1, int(x + width - 1) + pad_x)
    y2 = min(image_height - 1, int(y + height - 1) + pad_y)
    return x1, y1, x2, y2


def _evaluate(
    runner: _GradCamRunner, resized_rgb: np.ndarray, tta: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one image, optionally averaging four flipped views."""
    base = resized_rgb.astype(np.float32, copy=False)
    if not tta:
        probabilities, cams = runner.run(base[np.newaxis, ...])
        return probabilities[0], cams[0]

    probability_views: list[np.ndarray] = []
    cam_views: list[np.ndarray] = []
    for flip_code in (None, 1, 0, -1):
        view = base if flip_code is None else cv2.flip(base, flip_code)
        probabilities, cams = runner.run(view[np.newaxis, ...])
        probability_views.append(probabilities[0])
        cam = cams[0]
        if flip_code in (1, -1):
            cam = cv2.flip(cam, 1)
        if flip_code in (0, -1):
            cam = cv2.flip(cam, 0)
        cam_views.append(cam)
    return np.mean(probability_views, axis=0), np.mean(cam_views, axis=0)


def predict(
    frame_rgb: np.ndarray,
    model: tf.keras.Model | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    cam_threshold: float = CAM_THRESHOLD,
    tta: bool = False,
) -> dict[str, Any]:
    """Predict an RGB image and return its class, status, box, and Grad-CAM."""
    if model is None:
        model = _DEFAULT_MODEL
        if model is None:
            model = load_model_once(DEFAULT_MODEL_PATH)
    if (
        not isinstance(frame_rgb, np.ndarray)
        or frame_rgb.ndim != 3
        or frame_rgb.shape[2] != 3
        or frame_rgb.shape[0] == 0
        or frame_rgb.shape[1] == 0
    ):
        raise ValueError("frame_rgb must be a non-empty RGB image with shape (H, W, 3).")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not 0.0 <= cam_threshold <= 1.0:
        raise ValueError("cam_threshold must be between 0 and 1.")

    started = time.perf_counter()
    original_height, original_width = frame_rgb.shape[:2]
    resized_rgb = cv2.resize(
        frame_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA
    ).astype(np.float32, copy=False)
    probabilities, small_cam = _evaluate(_get_runner(model), resized_rgb, tta)
    cam = cv2.resize(
        small_cam.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    cam = np.clip(cam, 0.0, 1.0)

    class_index = int(np.argmax(probabilities))
    label = CLASS_NAMES[class_index]
    confidence = float(probabilities[class_index])
    status = "ok"
    box = None

    if _green_fraction(frame_rgb) < GREEN_FRACTION_THRESHOLD:
        status = "no_leaf"
    elif confidence < confidence_threshold:
        status = "low_confidence"
        label = "uncertain"
    elif label != "healthy":
        box = _extract_box(cam, cam_threshold)

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "box": box,
        "status": status,
        "inference_ms": (time.perf_counter() - started) * 1000.0,
        "cam": cam,
    }
