"""Streamlit interface for real-time apple-leaf disease detection."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from gradcam import (
    CAM_THRESHOLD,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    predict,
    load_model_once,
)

MODEL_PATH = Path(__file__).parent / "model" / "leaf_classifier_fp32.keras"
INFERENCE_LOCK = threading.Lock()
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "scab": (40, 40, 235),
    "rust": (20, 135, 245),
    "multiple_diseases": (190, 75, 190),
    "healthy": (60, 180, 75),
}
DISPLAY_NAMES: dict[str, str] = {
    "healthy": "Healthy",
    "multiple_diseases": "Multiple diseases",
    "rust": "Rust",
    "scab": "Scab",
    "uncertain": "Uncertain",
}

st.set_page_config(page_title="Leaf Health", page_icon="🌿", layout="wide")


@st.cache_resource
def get_model(path: str) -> Any:
    """Cache the loaded model for all Streamlit reruns."""
    return load_model_once(path)


def _get_model_or_show_error() -> tuple[Any | None, str | None]:
    """Load the model and show actionable setup or version-mismatch errors."""
    if not MODEL_PATH.is_file():
        message = (
            f"Model file not found: `{MODEL_PATH}`. Place "
            "`leaf_classifier_fp32.keras` in the app's `model` folder."
        )
        st.error(message)
        return None, message
    try:
        return get_model(str(MODEL_PATH)), None
    except Exception as exc:
        message = (
            "The model could not be loaded. Check that the TensorFlow version "
            f"matches the one used to save it. Details: {exc}"
        )
        st.error(message)
        return None, message


def _class_color(label: str) -> tuple[int, int, int]:
    """Select the configured BGR drawing color for a class."""
    return CLASS_COLORS_BGR.get(label, (235, 235, 235))


def draw_result(frame_rgb: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw the prediction box and an appropriate status label on an RGB image."""
    output = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    status = result["status"]
    if status == "low_confidence":
        text = "Move closer / better light"
        color = (0, 190, 255)
    elif status == "no_leaf":
        text = "No leaf in view"
        color = (0, 190, 255)
    else:
        label = DISPLAY_NAMES.get(result["label"], result["label"].title())
        text = f"{label} {result['confidence']:.0%}"
        color = _class_color(result["label"])
        box = result["box"]
        if box is not None:
            x1, y1, x2, y2 = (int(value) for value in box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = 12, 34
    cv2.rectangle(
        output,
        (x - 6, y - text_height - 8),
        (x + text_width + 6, y + baseline + 5),
        color,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        output,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def _heatmap_overlay(frame_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a JET-colored Grad-CAM map over the original RGB image."""
    heatmap_bgr = cv2.applyColorMap(
        np.uint8(np.clip(cam, 0.0, 1.0) * 255.0), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 0.55, heatmap_rgb, 0.45, 0)


def show_prediction(
    image_rgb: np.ndarray, result: dict[str, Any], source: str
) -> None:
    """Render a still-image result, probabilities, and Grad-CAM overlay."""
    st.subheader(f"{source}: {DISPLAY_NAMES.get(result['label'], result['label'])}")
    if result["status"] == "low_confidence":
        st.info("Move closer / improve the light, then try again.")
    elif result["status"] == "no_leaf":
        st.info("No leaf in view. Point the camera at an apple leaf.")
    else:
        st.caption(
            f"Confidence: {result['confidence']:.1%} · "
            f"Inference: {result['inference_ms']:.0f} ms"
        )

    image_col, cam_col = st.columns(2)
    with image_col:
        st.image(draw_result(image_rgb, result), caption="Prediction")
    with cam_col:
        st.image(
            _heatmap_overlay(image_rgb, result["cam"]),
            caption="Grad-CAM focus overlay",
        )

    st.markdown("**Class probabilities**")
    show_probabilities(result["probabilities"])
    st.caption(
        "Highlighted region shows where the model focused (Grad-CAM), "
        "not an exact lesion outline."
    )


def show_probabilities(probabilities: dict[str, float]) -> None:
    """Render all four class probabilities as progress bars."""
    for class_name in CLASS_NAMES:
        value = float(probabilities.get(class_name, 0.0))
        st.progress(
            min(1.0, max(0.0, value)),
            text=f"{DISPLAY_NAMES[class_name]} · {value:.1%}",
        )


class LiveVideoProcessor:
    """Throttle, smooth, and annotate webcam frames in the WebRTC thread."""

    def __init__(
        self,
        model: Any,
        confidence_threshold: float,
        cam_threshold: float,
        tta: bool,
        inference_interval: float = 0.25,
    ) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.cam_threshold = cam_threshold
        self.tta = tta
        self.inference_interval = inference_interval
        self._state_lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._error: str | None = None
        self._last_inference_at = 0.0
        self._last_frame_at: float | None = None
        self._fps = 0.0
        self._label_history: deque[str] = deque(maxlen=5)
        self._smoothed_box: np.ndarray | None = None
        self._smoothed_label: str | None = None

    def snapshot(self) -> tuple[dict[str, Any] | None, str | None, float]:
        """Read the latest result, error, and callback FPS safely."""
        with self._state_lock:
            result = None if self._latest is None else self._latest.copy()
            return result, self._error, self._fps

    def _smooth(self, result: dict[str, Any]) -> dict[str, Any]:
        """Apply a five-result majority vote and EMA to box coordinates."""
        smoothed = result.copy()
        label = result["label"]
        if result["status"] != "ok" or label not in CLASS_NAMES:
            return smoothed

        self._label_history.append(label)
        highest_count = max(Counter(self._label_history).values())
        voted_label = next(
            item for item in reversed(self._label_history)
            if self._label_history.count(item) == highest_count
        )
        smoothed["label"] = voted_label
        smoothed["confidence"] = result["probabilities"][voted_label]

        box = result["box"]
        if box is None:
            self._smoothed_box = None
        else:
            current_box = np.asarray(box, dtype=np.float32)
            if self._smoothed_box is None or self._smoothed_label != voted_label:
                self._smoothed_box = current_box
            else:
                self._smoothed_box = 0.65 * self._smoothed_box + 0.35 * current_box
            smoothed["box"] = tuple(int(round(value)) for value in self._smoothed_box)
        self._smoothed_label = voted_label
        return smoothed

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each frame while running inference at most four times/second."""
        now = time.monotonic()
        with self._state_lock:
            if self._last_frame_at is not None and now > self._last_frame_at:
                self._fps = 1.0 / (now - self._last_frame_at)
        self._last_frame_at = now

        frame_rgb = frame.to_ndarray(format="rgb24")
        if now - self._last_inference_at >= self.inference_interval:
            try:
                with INFERENCE_LOCK:
                    result = predict(
                        frame_rgb,
                        self.model,
                        confidence_threshold=self.confidence_threshold,
                        cam_threshold=self.cam_threshold,
                        tta=self.tta,
                    )
                result = self._smooth(result)
                with self._state_lock:
                    self._latest = result
                    self._error = None
                self._last_inference_at = time.monotonic()
            except Exception as exc:
                # Surface callback failures in the main Streamlit thread.
                with self._state_lock:
                    self._error = f"Inference failed: {exc}"
                self._last_inference_at = time.monotonic()

        latest, _, _ = self.snapshot()
        output_rgb = frame_rgb if latest is None else draw_result(frame_rgb, latest)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(output_bgr, format="bgr24")


def render_live_metrics(
    processor: LiveVideoProcessor | None,
    model_error: str | None,
) -> None:
    """Refresh live probabilities and performance metrics every half-second."""
    if model_error:
        st.error(model_error)
        return
    if processor is None:
        st.caption("Start the camera to see live predictions and probabilities.")
        return

    result, inference_error, fps = processor.snapshot()
    if inference_error:
        st.error(inference_error)
    if result is None:
        st.caption("Waiting for the first camera frame…")
        return

    metric_col, latency_col = st.columns(2)
    metric_col.metric("Video FPS", f"{fps:.1f}")
    latency_col.metric("Inference latency", f"{result['inference_ms']:.0f} ms")
    if result["status"] == "low_confidence":
        st.info("Move closer / better light")
    elif result["status"] == "no_leaf":
        st.info("No leaf in view")
    else:
        st.markdown(
            f"**{DISPLAY_NAMES[result['label']]} · {result['confidence']:.1%}**"
        )
    show_probabilities(result["probabilities"])


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0d1712 0%, #14231b 100%); }
    h1, h2, h3 { color: #e8f4eb; }
    [data-testid="stMetric"] {
      background: rgba(255,255,255,0.06); padding: 0.8rem; border-radius: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 Leaf Health")
st.write(
    "Check apple leaves for healthy, rust, scab, or multiple-disease patterns "
    "with a live camera, a photo, or an uploaded image."
)

with st.sidebar:
    st.header("Detection settings")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.3, 0.9, CONFIDENCE_THRESHOLD, 0.05
    )
    cam_threshold = st.slider("CAM threshold", 0.2, 0.8, CAM_THRESHOLD, 0.05)
    tta = st.checkbox("Test-time augmentation (slower)", value=False)
    st.divider()
    st.subheader("About")
    st.write("Model: `leaf_classifier_fp32.keras` (EfficientNetB4)")
    st.write("Classes: healthy, multiple_diseases, rust, scab")

live_tab, photo_tab, upload_tab = st.tabs(
    ("Live camera", "Take a photo", "Upload image")
)

with live_tab:
    video_col, metrics_col = st.columns([1.7, 1])
    model, model_error = _get_model_or_show_error()
    processor: LiveVideoProcessor | None = None
    if model is not None:
        with video_col:
            context = webrtc_streamer(
                key="leaf-live-camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: LiveVideoProcessor(
                    model,
                    confidence_threshold,
                    cam_threshold,
                    tta,
                ),
                media_stream_constraints={
                    "video": {
                        "facingMode": {"ideal": "environment"},
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                    },
                    "audio": False,
                },
                async_processing=False,
            )
            processor = context.video_processor
            if context.state.playing:
                st.caption("Live view runs inference at most every 250 ms.")
            else:
                st.caption(
                    "Allow camera access when prompted. Camera access requires "
                    "HTTPS or `http://localhost`."
                )

    with metrics_col:
        @st.fragment(run_every="500ms")
        def live_metrics_fragment() -> None:
            render_live_metrics(processor, model_error)

        live_metrics_fragment()

with photo_tab:
    photo = st.camera_input("Take a leaf photo")
    if photo is not None:
        model, _ = _get_model_or_show_error()
        if model is not None:
            try:
                image_rgb = np.asarray(Image.open(photo).convert("RGB"))
                with INFERENCE_LOCK:
                    result = predict(
                        image_rgb,
                        model,
                        confidence_threshold=confidence_threshold,
                        cam_threshold=cam_threshold,
                        tta=tta,
                    )
                show_prediction(image_rgb, result, "Photo result")
            except Exception as exc:
                st.error(f"Could not analyze the photo: {exc}")

with upload_tab:
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image", type=("jpg", "jpeg", "png")
    )
    if uploaded_file is not None:
        model, _ = _get_model_or_show_error()
        if model is not None:
            try:
                image_rgb = np.asarray(Image.open(uploaded_file).convert("RGB"))
                with INFERENCE_LOCK:
                    result = predict(
                        image_rgb,
                        model,
                        confidence_threshold=confidence_threshold,
                        cam_threshold=cam_threshold,
                        tta=tta,
                    )
                show_prediction(image_rgb, result, "Upload result")
            except Exception as exc:
                st.error(f"Could not analyze the uploaded image: {exc}")
