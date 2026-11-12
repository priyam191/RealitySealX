from PIL import Image

from ai_image_detector import AiImageDetectionResult
from cnn_detector import CnnDetectionResult
from ela_analyzer import analyze_image
from generate_sample_images import make_spliced
from result_analyzer import build_analysis_summary
from vit_detector import VitDetectionResult


def test_summary_is_incomplete_when_no_applicable_layer_ran() -> None:
    summary = build_analysis_summary(
        None,
        None,
        None,
        None,
        None,
    )

    assert summary.label == "Analysis incomplete"
    assert not summary.completed_layers
    assert len(summary.not_run_layers) == 4


def test_png_ela_is_not_assessed_as_a_suspiciousness_result() -> None:
    ela = analyze_image(Image.new("RGB", (32, 32), (20, 40, 60)), original_format="PNG")

    summary = build_analysis_summary(
        ela,
        "PNG",
        None,
        None,
        None,
    )

    assert summary.label == "Analysis incomplete"
    assert "ELA (PNG-to-JPEG conversion only)" in summary.not_assessed_layers
    assert not summary.completed_layers


def test_flagged_ela_is_reported_without_score_fusion() -> None:
    ela = analyze_image(make_spliced(), original_format="JPEG")

    summary = build_analysis_summary(
        ela,
        "JPEG",
        None,
        None,
        None,
    )

    assert summary.label == "Suspicious indication detected"
    assert summary.completed_layers == ("ELA",)
    assert len(summary.suspicious_indications) == 1


def test_ai_classifier_real_result_does_not_prove_authenticity() -> None:
    ai_result = AiImageDetectionResult(
        status="completed",
        model_name="test ViT",
        predicted_class="Real photograph",
        ai_generated_score=0.1,
        predicted_class_score=0.9,
        explanation="Test result.",
        input_note="Test input.",
    )

    summary = build_analysis_summary(
        None,
        "JPEG",
        None,
        None,
        ai_result,
    )

    assert summary.label == "No suspicious indication detected"
    assert "does not prove" in summary.explanation
    assert summary.completed_layers == ("AI-generated image detector",)


def test_face_models_with_disagreement_are_kept_separate() -> None:
    cnn = CnnDetectionResult(
        status="completed",
        model_name="CNN",
        prediction="Real",
        confidence=0.8,
        face_detected=True,
        face_crop=Image.new("RGB", (20, 20)),
        explanation="CNN result",
    )
    vit = VitDetectionResult(
        status="completed",
        model_name="ViT",
        prediction="Manipulated",
        confidence=0.7,
        real_score=0.3,
        manipulated_score=0.7,
        explanation="ViT result",
    )

    summary = build_analysis_summary(None, "JPEG", cnn, vit, None)

    assert summary.label == "Different models produced conflicting results"
    assert summary.completed_layers == (
        "CNN face manipulation detector",
        "Vision Transformer deepfake detector",
    )
