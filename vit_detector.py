"""Free ViT deepfake inference using a pinned public checkpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

MODEL_REPOSITORY = "dima806/deepfake_vs_real_image_detection"
MODEL_REVISION = "29e4cf9efc543845610045f6ba7e88e5cf9d9301"
MODEL_NAME = "ViT-Base (real/fake faces)"


class VitDetectorError(RuntimeError):
    """Raised when the ViT deepfake model cannot run."""


@dataclass(frozen=True)
class VitDetectionResult:
    status: Literal["completed"]
    model_name: str
    prediction: Literal["Real", "Manipulated"]
    confidence: float
    real_score: float
    manipulated_score: float
    explanation: str


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    try:
        import torch
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise VitDetectorError(
            "ViT dependencies are missing. Run `python -m pip install -r requirements.txt`."
        ) from exc

    local_path = os.environ.get("REALITYSEAL_VIT_MODEL_PATH")
    model_path = Path(local_path).expanduser() if local_path else MODEL_REPOSITORY
    if local_path and not model_path.is_dir():
        raise VitDetectorError(f"Configured ViT model directory does not exist: {model_path}")

    try:
        processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            revision=None if local_path else MODEL_REVISION,
            trust_remote_code=False,
        )
        model = AutoModelForImageClassification.from_pretrained(
            str(model_path),
            revision=None if local_path else MODEL_REVISION,
            use_safetensors=True,
            trust_remote_code=False,
        )
        model.to("cpu")
        model.eval()
    except (OSError, ValueError, RuntimeError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        raise VitDetectorError(
            "Could not load the free ViT checkpoint. Check your internet connection "
            "or set REALITYSEAL_VIT_MODEL_PATH to a local model directory."
        ) from exc

    labels = {
        int(index): str(label).strip().lower()
        for index, label in model.config.id2label.items()
    }
    if labels != {0: "real", 1: "fake"}:
        raise VitDetectorError(
            f"The ViT checkpoint has an unexpected class mapping: {labels}."
        )
    return model, processor, torch


def analyze_vit_deepfake(image: Image.Image) -> VitDetectionResult:
    """Classify an uploaded image using the pinned fine-tuned face-deepfake ViT."""
    model, processor, torch = _load_model()
    rgb = ImageOps.exif_transpose(image).convert("RGB")

    try:
        inputs = processor(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())
        real_score = float(probabilities[0].item())
        manipulated_score = float(probabilities[1].item())
    except (RuntimeError, ValueError, IndexError, KeyError) as exc:
        raise VitDetectorError("The ViT failed while analyzing the uploaded image.") from exc

    if not np.isfinite([confidence, real_score, manipulated_score]).all():
        raise VitDetectorError("The ViT returned non-finite class scores.")

    return VitDetectionResult(
        status="completed",
        model_name=MODEL_NAME,
        prediction="Manipulated" if predicted_index == 1 else "Real",
        confidence=confidence,
        real_score=real_score,
        manipulated_score=manipulated_score,
        explanation=(
            "This free research ViT is fine-tuned on the OpenForensics dataset. "
            "Its model card reports 96.56% test accuracy, but that result has not "
            "been independently validated here and may not generalize to other "
            "datasets or newer manipulation methods. The score is a model output, "
            "not a calibrated probability or proof of authenticity."
        ),
    )
