from types import SimpleNamespace

from PIL import Image

import torch

import vit_detector
from vit_detector import VitDetectionResult, analyze_vit_deepfake


class _FakeProcessor:
    def __call__(self, images, return_tensors):
        assert images.mode == "RGB"
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


class _FakeModel:
    def __call__(self, **inputs):
        assert inputs["pixel_values"].shape == (1, 3, 224, 224)
        return SimpleNamespace(logits=torch.tensor([[0.0, 3.0]]))


def test_vit_returns_real_fake_prediction_and_scores(monkeypatch) -> None:
    monkeypatch.setattr(
        vit_detector,
        "_load_model",
        lambda: (_FakeModel(), _FakeProcessor(), torch),
    )

    result = analyze_vit_deepfake(Image.new("RGB", (500, 300)))

    assert result.status == "completed"
    assert result.prediction == "Manipulated"
    assert result.confidence > 0.9
    assert result.manipulated_score > result.real_score


def test_vit_result_contract_is_independent() -> None:
    result = VitDetectionResult(
        status="completed",
        model_name="test ViT",
        prediction="Real",
        confidence=0.8,
        real_score=0.8,
        manipulated_score=0.2,
        explanation="Test result.",
    )

    assert result.prediction == "Real"
