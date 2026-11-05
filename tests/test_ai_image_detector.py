import numpy as np
from PIL import Image

import ai_image_detector


def test_preprocess_image_returns_normalized_model_input() -> None:
    image = Image.new("RGB", (440, 440), (255, 0, 0))

    tensor = ai_image_detector.preprocess_image(image)

    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_preprocess_image_flattens_transparency_on_white() -> None:
    image = Image.new("RGBA", (440, 440), (0, 0, 0, 0))

    tensor = ai_image_detector.preprocess_image(image)

    expected_white = (1.0 - ai_image_detector._IMAGE_MEAN) / ai_image_detector._IMAGE_STD
    np.testing.assert_allclose(tensor[0, :, 100, 100], expected_white, atol=1e-6)


def test_attention_overlay_is_same_size_and_disclaims_proof() -> None:
    crop = Image.new("RGB", (384, 384), (120, 80, 40))
    attention = np.zeros((1, 6, 577, 577), dtype=np.float32)
    attention[:, :, 0, 1] = 1.0

    result = ai_image_detector._attention_overlay(crop, attention)

    assert result.overlay.size == crop.size
    assert "not" in result.explanation
    assert "proof" in result.explanation


def test_inference_returns_score_and_explanation_without_claiming_proof(monkeypatch) -> None:
    class FakeTensor:
        def reshape(self, *_):
            return self

        def __getitem__(self, _):
            return self

        def item(self):
            return 2.0

    class FakeTorch:
        @staticmethod
        def from_numpy(_):
            return object()

        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    monkeypatch.setattr(
        ai_image_detector,
        "_load_model",
        lambda: (lambda _: FakeTensor(), FakeTorch),
    )
    result = ai_image_detector.analyze_ai_generated_image(Image.new("RGB", (512, 512)))

    assert result.predicted_class == "AI-generated image"
    assert 0.5 < result.ai_generated_score < 1.0
    assert result.predicted_class_score == result.ai_generated_score
    assert "not proof" in result.explanation
