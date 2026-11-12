"""Independent-result summary for the combined analysis dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import config
from ai_image_detector import AiImageDetectionResult
from cnn_detector import CnnDetectionResult
from ela_analyzer import ElaResult
from vit_detector import VitDetectionResult


@dataclass(frozen=True)
class AnalysisSummary:
    label: Literal[
        "No suspicious indication detected",
        "Suspicious indication detected",
        "Analysis incomplete",
    ]
    explanation: str
    completed_layers: tuple[str, ...]
    suspicious_indications: tuple[str, ...]
    not_assessed_layers: tuple[str, ...]
    unavailable_models: tuple[str, ...]
    failed_layers: tuple[str, ...]


def build_analysis_summary(
    ela: ElaResult | None,
    image_format: str | None,
    cnn: CnnDetectionResult,
    vit: VitDetectionResult,
    ai_image: AiImageDetectionResult | None,
    ai_error: str | None = None,
) -> AnalysisSummary:
    completed_layers: list[str] = []
    suspicious_indications: list[str] = []
    not_assessed_layers: list[str] = []
    unavailable_models = [
        result.model_name
        for result in (cnn, vit)
        if result.status == "unavailable"
    ]
    failed_layers = ["AI-generated image detector"] if ai_error else []

    if ela is not None:
        if (image_format or "").upper() == "PNG":
            not_assessed_layers.append("ELA (PNG-to-JPEG conversion only)")
        else:
            completed_layers.append("ELA")
            if ela.suspicion.score > config.LOW_SCORE_MAX:
                suspicious_indications.append(
                    f"ELA heuristic flagged an unusual compression-residual pattern "
                    f"(score {ela.suspicion.score}/100)."
                )

    if ai_image is not None:
        completed_layers.append("AI-generated image detector")
        if ai_image.predicted_class == "AI-generated image":
            suspicious_indications.append(
                "The AI-generated-image detector classified the image as "
                "AI-generated; this is not evidence of facial manipulation."
            )

    if not completed_layers:
        label = "Analysis incomplete"
        explanation = (
            "No applicable analysis layer has completed. Run ELA on a JPEG or "
            "run the AI-generated-image detector. The CNN and facial-deepfake ViT "
            "models are unavailable."
        )
    elif suspicious_indications:
        label = "Suspicious indication detected"
        explanation = (
            "At least one completed analysis reported an indicator for closer "
            "review. This is not a determination that the image is fake or edited."
        )
    else:
        label = "No suspicious indication detected"
        explanation = (
            "The completed checks did not flag a suspicious indication. This does "
            "not prove the image is authentic or unedited."
        )

    return AnalysisSummary(
        label=label,
        explanation=explanation,
        completed_layers=tuple(completed_layers),
        suspicious_indications=tuple(suspicious_indications),
        not_assessed_layers=tuple(not_assessed_layers),
        unavailable_models=tuple(unavailable_models),
        failed_layers=tuple(failed_layers),
    )
