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
        "Different models produced conflicting results",
        "Analysis incomplete",
    ]
    explanation: str
    completed_layers: tuple[str, ...]
    suspicious_indications: tuple[str, ...]
    not_assessed_layers: tuple[str, ...]
    not_run_layers: tuple[str, ...]
    unavailable_models: tuple[str, ...]
    failed_layers: tuple[str, ...]


def build_analysis_summary(
    ela: ElaResult | None,
    image_format: str | None,
    cnn: CnnDetectionResult | None,
    vit: VitDetectionResult | None,
    ai_image: AiImageDetectionResult | None,
    cnn_error: str | None = None,
    vit_error: str | None = None,
    ai_error: str | None = None,
) -> AnalysisSummary:
    completed_layers: list[str] = []
    suspicious_indications: list[str] = []
    not_assessed_layers: list[str] = []
    not_run_layers: list[str] = []
    failed_layers: list[str] = []

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
    else:
        not_run_layers.append("ELA")

    if cnn is None:
        (failed_layers if cnn_error else not_run_layers).append(
            "CNN face manipulation detector"
        )
    elif cnn.status == "completed":
        completed_layers.append("CNN face manipulation detector")
        if cnn.prediction == "Manipulated":
            score_text = (
                f" (model score {cnn.confidence:.1%})"
                if cnn.confidence is not None
                else ""
            )
            suspicious_indications.append(
                f"The CNN classified the detected face as manipulated{score_text}."
            )
    else:
        not_assessed_layers.append("CNN face manipulation detector (no face detected)")

    if vit is None:
        (failed_layers if vit_error else not_run_layers).append(
            "Vision Transformer deepfake detector"
        )
    else:
        completed_layers.append("Vision Transformer deepfake detector")
        if vit.prediction == "Manipulated":
            score_text = (
                f" (model score {vit.confidence:.1%})"
                if vit.confidence is not None
                else ""
            )
            suspicious_indications.append(
                f"The ViT classified the image as manipulated{score_text}."
            )

    if ai_image is None:
        (failed_layers if ai_error else not_run_layers).append(
            "AI-generated image detector"
        )
    else:
        completed_layers.append("AI-generated image detector")
        if ai_image.predicted_class == "AI-generated image":
            suspicious_indications.append(
                "The AI-generated-image detector classified the image as "
                "AI-generated; this is a separate task from facial manipulation."
            )

    detectors_disagree = (
        cnn is not None
        and cnn.status == "completed"
        and vit is not None
        and cnn.prediction != vit.prediction
    )
    if detectors_disagree:
        label = "Different models produced conflicting results"
        explanation = (
            "The CNN and ViT produced different face-manipulation classifications. "
            "Review both outputs independently; neither result is proof."
        )
    elif not completed_layers:
        label = "Analysis incomplete"
        explanation = (
            "No applicable analysis layer has completed. Run one or more "
            "independent analysis modules."
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
        not_run_layers=tuple(not_run_layers),
        unavailable_models=(),
        failed_layers=tuple(failed_layers),
    )
