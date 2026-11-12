from PIL import Image

from ai_image_detector import AiImageDetectionResult
from cnn_detector import analyze_face_manipulation
from ela_analyzer import analyze_image
from generate_sample_images import make_spliced
from result_analyzer import build_analysis_summary
from vit_detector import analyze_vit_deepfake


def test_summary_is_incomplete_when_no_applicable_layer_ran() -> None:
    summary = build_analysis_summary(
        None,
        None,
        analyze_face_manipulation(),
        analyze_vit_deepfake(),
        None,
    )

    assert summary.label == "Analysis incomplete"
    assert not summary.completed_layers
    assert len(summary.unavailable_models) == 2


def test_png_ela_is_not_assessed_as_a_suspiciousness_result() -> None:
    ela = analyze_image(Image.new("RGB", (32, 32), (20, 40, 60)), original_format="PNG")

    summary = build_analysis_summary(
        ela,
        "PNG",
        analyze_face_manipulation(),
        analyze_vit_deepfake(),
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
        analyze_face_manipulation(),
        analyze_vit_deepfake(),
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
        analyze_face_manipulation(),
        analyze_vit_deepfake(),
        ai_result,
    )

    assert summary.label == "No suspicious indication detected"
    assert "does not prove" in summary.explanation
    assert summary.completed_layers == ("AI-generated image detector",)
