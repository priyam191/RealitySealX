"""Free EfficientNet-B2 face-manipulation inference with local face detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

MODEL_REPOSITORY = "ThinothW/DFDS-EfficientNet-B2-Deepfake-Detection-Engine"
MODEL_REVISION = "d7887b76ac0a12ef6a5109515cc1ee435caf8804"
MODEL_NAME = "DFDS EfficientNet-B2 (face crop)"
INPUT_SIZE = 260
MIN_FACE_SIZE = 50
FACE_PADDING = 20
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


class CnnDetectorError(RuntimeError):
    """Raised when the face detector or CNN model cannot run."""


@dataclass(frozen=True)
class CnnDetectionResult:
    status: Literal["completed", "not_applicable"]
    model_name: str
    prediction: Literal["Real", "Manipulated"] | None
    confidence: float | None
    face_detected: bool
    face_crop: Image.Image | None
    explanation: str


@lru_cache(maxsize=1)
def _face_cascade():
    try:
        import cv2
    except ImportError as exc:
        raise CnnDetectorError(
            "OpenCV is missing. Install the project requirements to enable face detection."
        ) from exc

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise CnnDetectorError(f"OpenCV face detector could not load: {cascade_path}")
    return cascade, cv2


def _largest_face_crop(image: Image.Image) -> Image.Image | None:
    cascade, cv2 = _face_cascade()
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    if len(boxes) == 0:
        return None

    x, y, width, height = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
    left = max(0, int(x) - FACE_PADDING)
    top = max(0, int(y) - FACE_PADDING)
    right = min(rgb.shape[1], int(x + width) + FACE_PADDING)
    bottom = min(rgb.shape[0], int(y + height) + FACE_PADDING)
    return Image.fromarray(rgb[top:bottom, left:right]).convert("RGB")


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any]:
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from safetensors import SafetensorError
        from torchvision import models
        from torch import nn
    except ImportError as exc:
        raise CnnDetectorError(
            "CNN dependencies are missing. Run `python -m pip install -r requirements.txt`."
        ) from exc

    local_path = os.environ.get("REALITYSEAL_CNN_MODEL_PATH")
    if local_path:
        checkpoint_path = Path(local_path).expanduser()
        if not checkpoint_path.is_file():
            raise CnnDetectorError(
                f"Configured CNN checkpoint does not exist: {checkpoint_path}"
            )
    else:
        try:
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPOSITORY,
                    filename="pytorch_model.bin",
                    revision=MODEL_REVISION,
                )
            )
        except (OSError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
            raise CnnDetectorError(
                "Could not download the free CNN checkpoint. Check your internet "
                "connection or set REALITYSEAL_CNN_MODEL_PATH to a local checkpoint."
            ) from exc

    try:
        model = models.efficientnet_b2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1408, 2),
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, torch
    except (OSError, RuntimeError, ValueError, KeyError, SafetensorError) as exc:
        raise CnnDetectorError(
            "The CNN checkpoint is invalid or does not match the documented "
            "EfficientNet-B2 architecture."
        ) from exc


def _preprocess(face: Image.Image):
    try:
        import torch
        from torchvision.transforms import functional as transforms
    except ImportError as exc:
        raise CnnDetectorError(
            "CNN dependencies are missing. Run `python -m pip install -r requirements.txt`."
        ) from exc

    resized = face.resize(
        (INPUT_SIZE, INPUT_SIZE),
        resample=Image.Resampling.BICUBIC,
    )
    tensor = transforms.to_tensor(resized)
    tensor = transforms.normalize(tensor, _IMAGE_MEAN, _IMAGE_STD)
    return tensor.unsqueeze(0)


def analyze_face_manipulation(image: Image.Image) -> CnnDetectionResult:
    """Detect the largest frontal face and classify its crop with the CNN."""
    face_crop = _largest_face_crop(image)
    if face_crop is None:
        return CnnDetectionResult(
            status="not_applicable",
            model_name=MODEL_NAME,
            prediction=None,
            confidence=None,
            face_detected=False,
            face_crop=None,
            explanation=(
                "Not Applicable - No Face Detected. The local frontal-face "
                "detector found no suitable face; no CNN prediction was made."
            ),
        )

    model, torch = _load_model()
    try:
        with torch.inference_mode():
            logits = model(_preprocess(face_crop))
            probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())
    except (RuntimeError, ValueError, IndexError) as exc:
        raise CnnDetectorError("The EfficientNet-B2 CNN failed during inference.") from exc

    if len(probabilities) != 2 or not np.isfinite(confidence):
        raise CnnDetectorError("The CNN returned invalid class scores.")

    return CnnDetectionResult(
        status="completed",
        model_name=MODEL_NAME,
        prediction="Manipulated" if predicted_index == 0 else "Real",
        confidence=confidence,
        face_detected=True,
        face_crop=face_crop,
        explanation=(
            "This free research model is trained on a curated face-manipulation "
            "dataset. Its authors report 67.95% accuracy on their cross-dataset "
            "evaluation; this result has not been independently validated here. "
            "It may perform poorly on unseen manipulation methods, and Haar "
            "face-crop preprocessing differs from the model authors' RetinaFace "
            "preprocessing. Treat this as a screening indicator, not proof."
        ),
    )
