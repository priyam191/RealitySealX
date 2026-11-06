"""Inference for Community Forensics AI-generated-image classification."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

MODEL_REPOSITORY = "OwensLab/commfor-model-384"
MODEL_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
MODEL_NAME = "Community Forensics ViT-Small (384px)"
MODEL_INPUT_SIZE = 384
MODEL_RESIZE_SIZE = 440

_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AiImageDetectorError(RuntimeError):
    """Raised when the AI-generated-image detector cannot run."""


@dataclass(frozen=True)
class AiImageDetectionResult:
    status: Literal["completed"]
    model_name: str
    predicted_class: Literal["Real photograph", "AI-generated image"]
    ai_generated_score: float
    predicted_class_score: float
    explanation: str
    input_note: str


def _rgb_on_white(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply the official model's resize, center-crop, and ImageNet normalization."""
    rgb = _rgb_on_white(image)
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise AiImageDetectorError("The image has invalid dimensions.")

    scale = MODEL_RESIZE_SIZE / min(width, height)
    resized_size = (int(width * scale), int(height * scale))
    resized = rgb.resize(resized_size, resample=Image.Resampling.BILINEAR)

    left = (resized.width - MODEL_INPUT_SIZE) // 2
    top = (resized.height - MODEL_INPUT_SIZE) // 2
    cropped = resized.crop(
        (left, top, left + MODEL_INPUT_SIZE, top + MODEL_INPUT_SIZE)
    )
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    normalized = (pixels - _IMAGE_MEAN) / _IMAGE_STD
    return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)


def _resolve_checkpoint() -> Path:
    configured_path = os.environ.get("REALITYSEAL_AI_MODEL_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_file():
            raise AiImageDetectorError(
                f"Configured AI detector checkpoint does not exist: {path}"
            )
        return path

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        return Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename="model.safetensors",
                revision=MODEL_REVISION,
            )
        )
    except ImportError as exc:
        raise AiImageDetectorError(
            "The AI detector dependencies are missing. Install requirements.txt."
        ) from exc
    except (OSError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        raise AiImageDetectorError(
            "Could not retrieve the Community Forensics checkpoint. Check the "
            "network connection or set REALITYSEAL_AI_MODEL_PATH to a local "
            "compatible checkpoint."
        ) from exc


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any]:
    try:
        import timm
        import torch
        from safetensors.torch import load_file
        from safetensors import SafetensorError
        from torch import nn
    except ImportError as exc:
        raise AiImageDetectorError(
            "The AI detector dependencies are missing. Install requirements.txt."
        ) from exc

    checkpoint = _resolve_checkpoint()

    class CommunityForensicsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vit = timm.create_model(
                "vit_small_patch16_384.augreg_in21k_ft_in1k",
                pretrained=False,
            )
            self.vit.head = nn.Linear(384, 1)

        def forward(self, tensor):
            return self.vit(tensor)

    try:
        model = CommunityForensicsModel()
        state_dict = load_file(str(checkpoint), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to("cpu")
        model.eval()
        return model, torch
    except (OSError, RuntimeError, ValueError, KeyError, SafetensorError) as exc:
        raise AiImageDetectorError(
            "The configured checkpoint could not be loaded with the Community "
            "Forensics ViT-Small architecture."
        ) from exc


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def analyze_ai_generated_image(image: Image.Image) -> AiImageDetectionResult:
    """Classify a general image with the pinned Community Forensics checkpoint."""
    tensor = preprocess_image(image)
    model, torch = _load_model()

    try:
        input_tensor = torch.from_numpy(tensor)
        with torch.inference_mode():
            output = model(input_tensor)
        logit = float(output.reshape(-1)[0].item())
    except (RuntimeError, ValueError, IndexError) as exc:
        raise AiImageDetectorError(
            "The Community Forensics model failed during CPU inference."
        ) from exc

    if not math.isfinite(logit):
        raise AiImageDetectorError("The model returned a non-finite score.")

    ai_score = _sigmoid(logit)
    predicted_class = (
        "AI-generated image" if ai_score >= 0.5 else "Real photograph"
    )
    class_score = ai_score if ai_score >= 0.5 else 1.0 - ai_score
    short_edge = min(image.size)
    input_note = (
        "The model resizes the shorter edge to 440 pixels and center-crops to "
        "384×384, so content near image edges may be omitted."
    )
    if short_edge < MODEL_INPUT_SIZE:
        input_note += (
            " The source is smaller than the model crop and will be upscaled; "
            "low-detail inputs may be less reliable."
        )

    return AiImageDetectionResult(
        status="completed",
        model_name=MODEL_NAME,
        predicted_class=predicted_class,
        ai_generated_score=ai_score,
        predicted_class_score=class_score,
        explanation=(
            f"The model's AI-generated score is {ai_score:.3f}. This is a screening "
            "signal from a model trained on the Community Forensics distribution, "
            "not proof of origin or a calibrated probability. Results may differ "
            "for unseen generators, edits, or heavily processed images."
        ),
        input_note=input_note,
    )
