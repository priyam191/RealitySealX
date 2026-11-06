"""Availability contract for the CNN facial-manipulation detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CnnDetectionResult:
    status: Literal["unavailable"]
    model_name: str
    prediction: None
    confidence: None
    face_detected: None
    explanation: str


def analyze_face_manipulation() -> CnnDetectionResult:
    """Report that inference is unavailable until a compatible model is configured."""
    return CnnDetectionResult(
        status="unavailable",
        model_name="Xception CNN (checkpoint not configured)",
        prediction=None,
        confidence=None,
        face_detected=None,
        explanation=(
            "No compatible, licensed, and verified CNN checkpoint is configured. "
            "No face detection or manipulation prediction was performed."
        ),
    )
