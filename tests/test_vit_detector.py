from vit_detector import analyze_vit_deepfake


def test_unconfigured_vit_does_not_emit_a_prediction() -> None:
    result = analyze_vit_deepfake()

    assert result.status == "unavailable"
    assert result.model_name
    assert result.prediction is None
    assert result.confidence is None
    assert "No deepfake prediction was performed" in result.explanation
