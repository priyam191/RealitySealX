"""Inference for Community Forensics AI-generated-image classification."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

MODEL_REPOSITORY = "OwensLab/commfor-model-384"
MODEL_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
MODEL_NAME = "Community Forensics ViT-Small (384px)"
MODEL_INPUT_SIZE = 384
MODEL_RESIZE_SIZE = 440

_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL_INFERENCE_LOCK = threading.Lock()


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


@dataclass(frozen=True)
class AttentionVisualization:
    overlay: Image.Image
    explanation: str


def _rgb_on_white(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _prepare_model_crop(image: Image.Image) -> Image.Image:
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
    return cropped


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply the official model's resize, center-crop, and ImageNet normalization."""
    cropped = _prepare_model_crop(image)
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


def _attention_overlay(
    crop: Image.Image,
    attention_weights: np.ndarray,
) -> AttentionVisualization:
    if attention_weights.ndim != 4 or attention_weights.shape[0] != 1:
        raise AiImageDetectorError("The ViT returned an unsupported attention shape.")
    class_attention = attention_weights[0, :, 0, 1:].mean(axis=0)
    grid_size = math.isqrt(class_attention.size)
    if grid_size * grid_size != class_attention.size:
        raise AiImageDetectorError("The ViT attention tokens do not form a square grid.")

    attention_map = class_attention.reshape(grid_size, grid_size)
    minimum = float(np.min(attention_map))
    maximum = float(np.max(attention_map))
    if maximum > minimum:
        attention_map = (attention_map - minimum) / (maximum - minimum)
    else:
        attention_map = np.zeros_like(attention_map)

    gray_map = Image.fromarray(
        np.clip(attention_map * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    ).resize(crop.size, resample=Image.Resampling.BILINEAR)
    import matplotlib

    heatmap_array = matplotlib.colormaps["inferno"](
        np.asarray(gray_map, dtype=np.float32) / 255.0
    )[:, :, :3]
    heatmap = Image.fromarray((heatmap_array * 255.0).astype(np.uint8), mode="RGB")
    return AttentionVisualization(
        overlay=Image.blend(crop.convert("RGB"), heatmap, alpha=0.45),
        explanation=(
            "This overlay shows the final ViT layer's average attention from the "
            "classification token to image patches in the model's center crop. "
            "Attention is a model-focus visualization, not a manipulated-region "
            "map, causal explanation, or proof."
        ),
    )


def analyze_ai_generated_image(image: Image.Image) -> AiImageDetectionResult:
    """Classify a general image with the pinned Community Forensics checkpoint."""
    tensor = preprocess_image(image)
    model, torch = _load_model()

    try:
        input_tensor = torch.from_numpy(tensor)
        with _MODEL_INFERENCE_LOCK:
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


def generate_attention_visualization(image: Image.Image) -> AttentionVisualization:
    """Visualize last-layer CLS-to-patch attention for the AI detector's ViT."""
    crop = _prepare_model_crop(image)
    tensor = preprocess_image(image)
    model, torch = _load_model()
    try:
        attention_layer = model.vit.blocks[-1].attn
    except (AttributeError, IndexError) as exc:
        raise AiImageDetectorError(
            "The Community Forensics model does not expose a ViT attention layer."
        ) from exc
    attention_dropout = getattr(attention_layer, "attn_drop", None)
    if not hasattr(attention_layer, "fused_attn") or attention_dropout is None:
        raise AiImageDetectorError("Attention visualization is not supported by this ViT.")

    captured_attention: list[np.ndarray] = []

    def capture_attention(_module, _inputs, output) -> None:
        captured_attention.append(output.detach().cpu().numpy())

    with _MODEL_INFERENCE_LOCK:
        hook = None
        fused_attention = attention_layer.fused_attn
        try:
            attention_layer.fused_attn = False
            hook = attention_dropout.register_forward_hook(capture_attention)
            with torch.inference_mode():
                model(torch.from_numpy(tensor))
        except (RuntimeError, ValueError, IndexError, AttributeError) as exc:
            raise AiImageDetectorError(
                "Could not capture attention from the Community Forensics ViT."
            ) from exc
        finally:
            if hook is not None:
                hook.remove()
            attention_layer.fused_attn = fused_attention

    if not captured_attention:
        raise AiImageDetectorError("The ViT did not expose attention weights.")
    return _attention_overlay(crop, captured_attention[-1])
