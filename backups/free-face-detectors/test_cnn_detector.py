from cnn_detector import analyze_face_manipulation


def test_unconfigured_cnn_does_not_emit_a_prediction() -> None:
    result = analyze_face_manipulation()

    assert result.status == "unavailable"
    assert result.model_name
    assert result.prediction is None
    assert result.confidence is None
    assert result.face_detected is None
    assert "No face detection or manipulation prediction was performed" in result.explanation
