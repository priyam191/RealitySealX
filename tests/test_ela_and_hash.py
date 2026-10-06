from __future__ import annotations

import io

import numpy as np
from PIL import Image

import config
from ela_analyzer import (
    analyze_image,
    compute_ela_difference,
    compute_statistics,
    compute_suspicion_score,
    recompress_as_jpeg,
)
from image_io import ImageLoadError, load_image_from_bytes
from metadata_analyzer import sha256_bytes


def _solid_jpeg(color: tuple[int, int, int], size=(64, 48), quality=90) -> Image.Image:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def test_sha256_known_vector() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_changes_with_content() -> None:
    assert sha256_bytes(b"one") != sha256_bytes(b"two")


def test_recompress_same_size() -> None:
    original = _solid_jpeg((120, 80, 40))
    recompressed = recompress_as_jpeg(original, quality=90)
    assert recompressed.size == original.size
    assert recompressed.mode == "RGB"


def test_ela_difference_non_negative() -> None:
    original = _solid_jpeg((10, 20, 30), quality=95)
    recompressed = recompress_as_jpeg(original, quality=90)
    diff = compute_ela_difference(original, recompressed)
    assert diff.shape == (48, 64, 3)
    assert np.all(diff >= 0)


def test_statistics_keys_and_ranges() -> None:
    original = _solid_jpeg((200, 10, 10), quality=85)
    result = analyze_image(original, jpeg_quality=90, amplification_scale=10, hot_pixel_threshold=20)
    stats = result.statistics
    assert stats.mean_difference >= 0
    assert stats.max_difference >= stats.mean_difference
    assert stats.std_difference >= 0
    assert 0 <= stats.percent_pixels_above_threshold <= 100
    assert 0 <= result.suspicion.score <= 100
    assert result.ela_heatmap.size == original.size


def test_suspicion_score_increases_with_hot_pixels() -> None:
    from ela_analyzer import ElaStatistics

    low = ElaStatistics(
        mean_difference=1.0,
        max_difference=5.0,
        std_difference=0.5,
        percent_pixels_above_threshold=0.1,
        hot_pixel_threshold=20.0,
        pixel_count=100,
        p95_difference=2.0,
        p99_difference=3.0,
        block_contrast_ratio=1.1,
        outlier_block_percent=0.0,
    )
    high = ElaStatistics(
        mean_difference=config.SCORE_MEAN_CAP,
        max_difference=config.SCORE_MAX_CAP,
        std_difference=config.SCORE_STD_CAP,
        percent_pixels_above_threshold=config.SCORE_HOT_PCT_CAP,
        hot_pixel_threshold=20.0,
        pixel_count=100,
        p95_difference=config.SCORE_P99_CAP,
        p99_difference=config.SCORE_P99_CAP,
        block_contrast_ratio=1.0 + config.SCORE_CONTRAST_CAP,
        outlier_block_percent=config.SCORE_OUTLIER_BLOCK_CAP,
    )
    assert compute_suspicion_score(low).score < compute_suspicion_score(high).score
    assert compute_suspicion_score(high).score == 100


def test_png_note_mentions_jpeg_recompression() -> None:
    png_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (8, 9, 10)).save(png_buffer, format="PNG")
    png_buffer.seek(0)
    png = Image.open(png_buffer)
    png.load()
    result = analyze_image(png, original_format="PNG")
    assert "JPEG-recompression" in result.source_mode_note
    assert "PNG" in result.suspicion.simple_headline


def test_synthetic_splice_scores_higher_than_unedited() -> None:
    from generate_sample_images import make_recompressed, make_spliced, make_unedited

    unedited = analyze_image(make_unedited(), original_format="JPEG")
    spliced = analyze_image(make_spliced(), original_format="JPEG")
    recompressed = analyze_image(
        make_recompressed(make_unedited()),
        original_format="JPEG",
    )
    assert spliced.suspicion.score >= unedited.suspicion.score + 20
    assert spliced.suspicion.score > config.LOW_SCORE_MAX
    assert unedited.suspicion.score <= config.LOW_SCORE_MAX
    assert recompressed.suspicion.score < spliced.suspicion.score
    assert spliced.statistics.block_contrast_ratio > unedited.statistics.block_contrast_ratio
    assert unedited.statistics.block_contrast_ratio < 5.0
    assert unedited.statistics.outlier_block_percent < config.SCORE_OUTLIER_BLOCK_CAP
    assert spliced.statistics.hotspot_bbox is not None


def test_shipped_sample_images_show_distinct_ela_results() -> None:
    with Image.open(config.SAMPLE_IMAGES_DIR / "01_unedited.jpg") as image:
        unedited = analyze_image(image, original_format="JPEG")
    with Image.open(config.SAMPLE_IMAGES_DIR / "02_spliced.jpg") as image:
        spliced = analyze_image(image, original_format="JPEG")
    assert unedited.suspicion.score <= config.LOW_SCORE_MAX
    assert spliced.suspicion.score > config.LOW_SCORE_MAX
    assert spliced.suspicion.score > unedited.suspicion.score


def test_plain_language_fields_present() -> None:
    result = analyze_image(_solid_jpeg((200, 10, 10), quality=85))
    assert result.suspicion.simple_headline
    assert result.suspicion.simple_status
    assert result.overlay_image.size == result.ela_heatmap.size
    assert "not proof" in result.suspicion.simple_explanation


def test_png_plain_language_result_explains_conversion_limit() -> None:
    result = analyze_image(Image.new("RGB", (32, 32), (8, 9, 10)), original_format="PNG")
    assert result.suspicion.simple_status == "Not assessed for suspicious editing"
    assert "cannot determine whether the PNG was edited" in result.suspicion.simple_explanation
    assert result.suspicion.indicator == "Not assessed (PNG-to-JPEG conversion only)"


def test_flagged_wording_matches_score_bands() -> None:
    from ela_analyzer import ElaStatistics

    low = ElaStatistics(0.1, 1, 0.1, 0, 8, 100, block_contrast_ratio=1)
    moderate = ElaStatistics(3, 30, 4, 8, 8, 100, p99_difference=12, block_contrast_ratio=2)
    high = ElaStatistics(
        6, 60, 8, 12, 8, 100, p99_difference=18,
        block_contrast_ratio=4.5, outlier_block_percent=12,
    )
    assert compute_suspicion_score(low).indicator == "Not flagged"
    assert compute_suspicion_score(moderate).indicator == (
        "Flagged as suspicious (review recommended)"
    )
    assert compute_suspicion_score(high).indicator == "Flagged as highly suspicious"


def test_report_exposes_presentation_score_and_flagged_status() -> None:
    from metadata_analyzer import analyze_file
    from report_generator import build_report_payload

    image = _solid_jpeg((200, 10, 10), quality=85)
    ela = analyze_image(image, original_format="JPEG")
    meta = analyze_file("test.jpg", b"test image bytes", image)
    report = build_report_payload(meta, ela)
    assert report["ela_contrast_score"] == ela.suspicion.score
    assert report["ela_score_label"] == "ELA contrast score"
    assert report["flagged"] == (ela.suspicion.score > config.LOW_SCORE_MAX)

    png_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (8, 9, 10)).save(png_buffer, format="PNG")
    png_buffer.seek(0)
    png = Image.open(png_buffer)
    png.load()
    png_ela = analyze_image(png, original_format="PNG")
    png_meta = analyze_file("test.png", b"test PNG bytes", png)
    png_report = build_report_payload(png_meta, png_ela)
    assert png_report["ela_score_label"] == "JPEG conversion difference"
    assert png_report["flagged"] is False


def test_load_rejects_bad_extension() -> None:
    try:
        load_image_from_bytes(b"not-an-image", "notes.txt")
        assert False, "expected ImageLoadError"
    except ImageLoadError:
        pass


def test_load_rejects_corrupt_jpeg_payload() -> None:
    try:
        load_image_from_bytes(b"this is not a jpeg", "broken.jpg")
        assert False, "expected ImageLoadError"
    except ImageLoadError:
        pass


def test_png_note_mentions_jpeg_recompression() -> None:
    png = Image.new("RGB", (32, 32), (8, 9, 10))
    result = analyze_image(png, original_format="PNG")
    assert "JPEG-recompression" in result.source_mode_note
