from PIL import Image

import cnn_detector
from cnn_detector import analyze_face_manipulation


def test_no_face_is_not_applicable_and_does_not_load_model(monkeypatch) -> None:
    monkeypatch.setattr(cnn_detector, "_largest_face_crop", lambda image: None)
    monkeypatch.setattr(
        cnn_detector,
        "_load_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should not be loaded")),
    )

    result = analyze_face_manipulation(Image.new("RGB", (64, 64), "white"))

    assert result.status == "not_applicable"
    assert result.prediction is None
    assert result.confidence is None
    assert result.face_detected is False
    assert "Not Applicable - No Face Detected" in result.explanation


def test_cnn_runs_on_detected_face_and_returns_class_score(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(
        cnn_detector,
        "_largest_face_crop",
        lambda image: Image.new("RGB", (100, 100), "white"),
    )
    fake_model = lambda tensor: torch.tensor([[2.0, 0.0]])
    monkeypatch.setattr(cnn_detector, "_load_model", lambda: (fake_model, torch))

    result = analyze_face_manipulation(Image.new("RGB", (200, 200), "white"))

    assert result.status == "completed"
    assert result.face_detected is True
    assert result.prediction == "Manipulated"
    assert result.confidence is not None and result.confidence > 0.8
    assert result.face_crop is not None
