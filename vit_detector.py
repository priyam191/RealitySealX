"""Availability contract for the Vision Transformer deepfake detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VitDetectionResult:
    status: Literal["unavailable"]
    model_name: str
    prediction: None
    confidence: None
    explanation: str


def analyze_vit_deepfake() -> VitDetectionResult:
    """Report that ViT inference is unavailable until a compatible model is approved."""
    return VitDetectionResult(
        status="unavailable",
        model_name="ViT deepfake detector (checkpoint not configured)",
        prediction=None,
        confidence=None,
        explanation=(
            "No compatible, documented, and verified ViT checkpoint is configured. "
            "No deepfake prediction was performed."
        ),
    )
