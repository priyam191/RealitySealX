"""
Error Level Analysis (ELA) for JPEG-style recompression comparison.

ELA saves a JPEG copy at a known quality, then measures the absolute
pixel difference between the RGB original and that recompressed copy.
Regions that were recently edited or saved at a different quality often
(but not always) show larger residuals.

This is a JPEG-recompression-based visualisation, not a definitive
forensic test.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import config


@dataclass
class ElaSettings:
    jpeg_quality: int
    amplification_scale: float
    hot_pixel_threshold: float


@dataclass
class ElaStatistics:
    mean_difference: float
    max_difference: float
    std_difference: float
    percent_pixels_above_threshold: float
    hot_pixel_threshold: float
    pixel_count: int
    p95_difference: float = 0.0
    p99_difference: float = 0.0
    block_contrast_ratio: float = 1.0
    outlier_block_percent: float = 0.0
    hotspot_bbox: tuple[int, int, int, int] | None = None


@dataclass
class SuspicionScore:
    score: int
    indicator: str
    interpretation: str
    components: dict[str, float]
    formula_note: str
    simple_headline: str = ""
    simple_status: str = ""
    simple_explanation: str = ""
    heatmap_hint: str = ""


@dataclass
class ElaResult:
    ela_rgb: Image.Image
    ela_heatmap: Image.Image
    overlay_image: Image.Image
    statistics: ElaStatistics
    suspicion: SuspicionScore
    settings: ElaSettings
    source_mode_note: str


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def recompress_as_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Write an in-memory JPEG copy and reopen it. Does not touch the original file."""
    buffer = io.BytesIO()
    _ensure_rgb(image).save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    recompressed.load()
    return recompressed.convert("RGB")


def compute_ela_difference(
    original_rgb: Image.Image,
    recompressed_rgb: Image.Image,
) -> np.ndarray:
    """Absolute per-channel difference as float32 array shaped (H, W, 3)."""
    orig = np.asarray(_ensure_rgb(original_rgb), dtype=np.float32)
    rec = np.asarray(_ensure_rgb(recompressed_rgb), dtype=np.float32)
    if orig.shape != rec.shape:
        raise ValueError("Original and recompressed images must have the same dimensions.")
    return np.abs(orig - rec)


def per_pixel_magnitude(abs_diff: np.ndarray) -> np.ndarray:
    """Mean absolute difference across RGB channels, range approximately 0–255."""
    return np.mean(abs_diff, axis=2)


def _block_means(magnitude: np.ndarray, block: int) -> np.ndarray:
    height, width = magnitude.shape
    rows = height // block
    cols = width // block
    if rows < 2 or cols < 2:
        return np.array([[float(np.mean(magnitude))]], dtype=np.float32)
    trimmed = magnitude[: rows * block, : cols * block]
    return trimmed.reshape(rows, block, cols, block).mean(axis=(1, 3))


def _hotspot_bbox_from_blocks(
    block_means: np.ndarray,
    median: float,
    block: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return (left, top, right, bottom) if a compact high-ELA region exists."""
    if block_means.size < 4:
        return None
    mad = float(np.median(np.abs(block_means - median))) + 1e-6
    threshold = max(
        median + config.BLOCK_OUTLIER_K * mad,
        median * config.BLOCK_OUTLIER_RATIO,
        median + config.HOTSPOT_MIN_DIFFERENCE,
    )
    mask = block_means >= threshold
    if not np.any(mask):
        return None

    visited = np.zeros(mask.shape, dtype=bool)
    largest_component: list[tuple[int, int]] = []
    for start_y, start_x in zip(*np.where(mask)):
        if visited[start_y, start_x]:
            continue
        component: list[tuple[int, int]] = []
        pending = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        while pending:
            y, x = pending.pop()
            component.append((y, x))
            for neighbor_y in range(max(0, y - 1), min(mask.shape[0], y + 2)):
                for neighbor_x in range(max(0, x - 1), min(mask.shape[1], x + 2)):
                    if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        pending.append((neighbor_y, neighbor_x))
        if len(component) > len(largest_component):
            largest_component = component
    if len(largest_component) < 4:
        return None

    ys, xs = zip(*largest_component)
    top = int(min(ys) * block)
    left = int(min(xs) * block)
    bottom = int(min(image_size[1], (max(ys) + 1) * block))
    right = int(min(image_size[0], (max(xs) + 1) * block))
    area = (right - left) * (bottom - top)
    image_area = image_size[0] * image_size[1]
    if area > 0.5 * image_area:
        return None
    return (left, top, right, bottom)


def compute_statistics(abs_diff: np.ndarray, hot_threshold: float) -> ElaStatistics:
    magnitude = per_pixel_magnitude(abs_diff)
    hot = magnitude > float(hot_threshold)
    block = int(config.BLOCK_SIZE)
    blocks = _block_means(magnitude, block)
    median_block = float(np.median(blocks))
    upper_block = float(np.percentile(blocks, config.BLOCK_CONTRAST_PERCENTILE))
    contrast = upper_block / max(median_block, config.BLOCK_CONTRAST_MIN_TYPICAL)
    mad = float(np.median(np.abs(blocks - median_block))) + 1e-6
    outlier_threshold = max(
        median_block + config.BLOCK_OUTLIER_K * mad,
        median_block * config.BLOCK_OUTLIER_RATIO,
        median_block + config.BLOCK_OUTLIER_MIN_DIFFERENCE,
    )
    outlier = blocks >= outlier_threshold
    height, width = magnitude.shape
    bbox = _hotspot_bbox_from_blocks(blocks, median_block, block, (width, height))
    return ElaStatistics(
        mean_difference=float(np.mean(magnitude)),
        max_difference=float(np.max(magnitude)),
        std_difference=float(np.std(magnitude)),
        percent_pixels_above_threshold=float(np.mean(hot) * 100.0),
        hot_pixel_threshold=float(hot_threshold),
        pixel_count=int(magnitude.size),
        p95_difference=float(np.percentile(magnitude, 95)),
        p99_difference=float(np.percentile(magnitude, 99)),
        block_contrast_ratio=float(contrast),
        outlier_block_percent=float(np.mean(outlier) * 100.0),
        hotspot_bbox=bbox,
    )


def amplify_for_display(abs_diff: np.ndarray, scale: float) -> Image.Image:
    """Scale residuals and clip to 8-bit RGB for a greyscale-style ELA view."""
    amplified = np.clip(abs_diff * float(scale), 0, 255).astype(np.uint8)
    return Image.fromarray(amplified, mode="RGB")


def magnitude_heatmap(abs_diff: np.ndarray, scale: float) -> Image.Image:
    """
    False-colour heatmap.

    Values are stretched using a robust percentile so a small pasted region
    remains visible even when the rest of the image has tiny residuals.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    magnitude = per_pixel_magnitude(abs_diff)
    amplified = magnitude * float(scale)
    p99 = float(np.percentile(amplified, 99.5))
    ceiling = max(p99, 8.0)
    normalised = np.clip(amplified / ceiling, 0.0, 1.0)
    coloured = (cm.inferno(normalised)[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(coloured, mode="RGB")


def draw_hotspot_overlay(original_rgb: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image:
    overlay = _ensure_rgb(original_rgb).copy()
    if bbox is None:
        return overlay
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = bbox
    for width in (6, 4, 2):
        colour = (255, 40, 40) if width == 6 else (255, 220, 80) if width == 4 else (255, 255, 255)
        draw.rectangle([left, top, right, bottom], outline=colour, width=width)
    draw.text((left + 6, max(0, top - 18)), "Area that stood out", fill=(255, 40, 40))
    return overlay


def _clip_unit(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return float(min(max(value, 0.0) / cap, 1.0))


def _plain_language(score: int, stats: ElaStatistics, original_format: str | None) -> tuple[str, str, str]:
    fmt = (original_format or "").upper()
    if fmt == "PNG":
        return (
            "PNG conversion check only",
            "JPEG conversion only - not an edit verdict",
            "In everyday terms: this file is PNG, not JPEG. The heatmap can look "
            "active simply because we converted it to JPEG for the test. Look at "
            "whether one patch is much brighter than the rest. A uniformly bright "
            "map on a PNG is not, by itself, proof that a photo was edited. "
            "This conversion check cannot determine whether the PNG was edited.",
        )
    if score <= config.LOW_SCORE_MAX:
        extra = ""
        if stats.block_contrast_ratio < 2.0:
            extra = " No single region stood out from the rest."
        return (
            config.SIMPLE_HEADLINE_LOW,
            config.SIMPLE_STATUS_LOW,
            config.SIMPLE_BODY_LOW + extra,
        )
    if score <= config.MODERATE_SCORE_MAX:
        return (
            config.SIMPLE_HEADLINE_MODERATE,
            config.SIMPLE_STATUS_MODERATE,
            config.SIMPLE_BODY_MODERATE,
        )
    return (
        config.SIMPLE_HEADLINE_HIGH,
        config.SIMPLE_STATUS_HIGH,
        config.SIMPLE_BODY_HIGH,
    )


def compute_suspicion_score(
    stats: ElaStatistics,
    original_format: str | None = None,
) -> SuspicionScore:
    """
    Transparent heuristic score in 0–100.

    Localisation terms (block contrast, outlier blocks, p99) are weighted more
    than the global mean so a small splice can raise the score. Block contrast
    uses an upper percentile and a 1.0-pixel floor to avoid division by near-zero
    background residuals.
    """
    mean_term = _clip_unit(stats.mean_difference, config.SCORE_MEAN_CAP)
    std_term = _clip_unit(stats.std_difference, config.SCORE_STD_CAP)
    max_term = _clip_unit(stats.max_difference, config.SCORE_MAX_CAP)
    hot_term = _clip_unit(stats.percent_pixels_above_threshold, config.SCORE_HOT_PCT_CAP)
    p99_term = _clip_unit(stats.p99_difference, config.SCORE_P99_CAP)
    contrast_term = _clip_unit(max(stats.block_contrast_ratio - 1.0, 0.0), config.SCORE_CONTRAST_CAP)
    outlier_term = _clip_unit(stats.outlier_block_percent, config.SCORE_OUTLIER_BLOCK_CAP)

    weighted = (
        config.WEIGHT_MEAN * mean_term
        + config.WEIGHT_STD * std_term
        + config.WEIGHT_MAX * max_term
        + config.WEIGHT_HOT * hot_term
        + config.WEIGHT_P99 * p99_term
        + config.WEIGHT_CONTRAST * contrast_term
        + config.WEIGHT_OUTLIER_BLOCKS * outlier_term
    )
    score = int(round(100.0 * float(np.clip(weighted, 0.0, 1.0))))

    if (original_format or "").upper() == "PNG":
        indicator = "JPEG conversion differences (not an edit verdict)"
        interpretation = (
            "This upload is PNG, so the ELA map comes from converting it to JPEG "
            "for comparison. Its score cannot be used to assess whether the PNG "
            "was edited."
        )
    elif score <= config.LOW_SCORE_MAX:
        indicator = "Low ELA anomaly"
        interpretation = config.INTERPRETATION_LOW
    elif score <= config.MODERATE_SCORE_MAX:
        indicator = "Moderate ELA anomaly"
        interpretation = config.INTERPRETATION_MODERATE
    else:
        indicator = "High ELA anomaly"
        interpretation = config.INTERPRETATION_HIGH

    headline, status, explanation = _plain_language(score, stats, original_format)

    components = {
        "mean_term_0_to_1": round(mean_term, 4),
        "std_term_0_to_1": round(std_term, 4),
        "max_term_0_to_1": round(max_term, 4),
        "hot_percent_term_0_to_1": round(hot_term, 4),
        "p99_term_0_to_1": round(p99_term, 4),
        "contrast_term_0_to_1": round(contrast_term, 4),
        "outlier_blocks_term_0_to_1": round(outlier_term, 4),
        "weight_mean": config.WEIGHT_MEAN,
        "weight_std": config.WEIGHT_STD,
        "weight_max": config.WEIGHT_MAX,
        "weight_hot": config.WEIGHT_HOT,
        "weight_p99": config.WEIGHT_P99,
        "weight_contrast": config.WEIGHT_CONTRAST,
        "weight_outlier_blocks": config.WEIGHT_OUTLIER_BLOCKS,
        "weighted_sum_0_to_1": round(float(weighted), 4),
        "block_contrast_ratio": round(stats.block_contrast_ratio, 4),
        "outlier_block_percent": round(stats.outlier_block_percent, 4),
        "p99_difference": round(stats.p99_difference, 4),
    }

    formula_note = (
        "Score = 100 × (w_mean·min(mean/MEAN_CAP,1) + w_std·min(std/STD_CAP,1) "
        "+ w_max·min(max/MAX_CAP,1) + w_hot·min(hot%/HOT_CAP,1) "
        "+ w_p99·min(p99/P99_CAP,1) + w_contrast·min((contrast-1)/CONTRAST_CAP,1) "
        "+ w_outlier·min(outlier_block%/OUTLIER_CAP,1)). "
        f"Block contrast is the {config.BLOCK_CONTRAST_PERCENTILE:g}th-percentile "
        f"block mean divided by the typical block mean (minimum denominator "
        f"{config.BLOCK_CONTRAST_MIN_TYPICAL:g}); outlier blocks must exceed the "
        f"typical value by at least {config.BLOCK_OUTLIER_MIN_DIFFERENCE:g}. "
        "Local contrast is weighted more than the image-wide average. "
        "Caps and weights are in config.py and require calibration on a labelled dataset."
    )
    return SuspicionScore(
        score=score,
        indicator=indicator,
        interpretation=interpretation,
        components=components,
        formula_note=formula_note,
        simple_headline=headline,
        simple_status=status,
        simple_explanation=explanation,
        heatmap_hint=config.HEATMAP_HINT,
    )


def source_mode_note(original_format: str | None) -> str:
    fmt = (original_format or "").upper()
    if fmt == "PNG":
        return (
            "The upload is PNG. ELA in this prototype works by converting the pixels "
            "to RGB, saving a JPEG copy at the configured quality, and comparing that "
            "copy with the RGB original. This is a JPEG-recompression-based analysis, "
            "not a native PNG forensic test, and it is not a definitive authenticity check."
        )
    if fmt in {"JPEG", "JPG"}:
        return (
            "The upload is JPEG. ELA recompresses a copy at the configured quality and "
            "compares it with the decoded original pixels. Residual patterns can reflect "
            "editing, prior recompression, or camera processing. This is not a definitive "
            "forensic test."
        )
    return (
        "ELA here is a JPEG-recompression comparison of RGB pixels. It is not a "
        "definitive forensic test of authenticity."
    )


def analyze_image(
    image: Image.Image,
    jpeg_quality: int | None = None,
    amplification_scale: float | None = None,
    hot_pixel_threshold: float | None = None,
    original_format: str | None = None,
) -> ElaResult:
    quality = int(jpeg_quality if jpeg_quality is not None else config.JPEG_RECOMPRESS_QUALITY)
    scale = float(
        amplification_scale if amplification_scale is not None else config.ELA_AMPLIFICATION_SCALE
    )
    hot = float(
        hot_pixel_threshold if hot_pixel_threshold is not None else config.HOT_PIXEL_THRESHOLD
    )

    original_rgb = _ensure_rgb(image)
    recompressed = recompress_as_jpeg(original_rgb, quality)
    abs_diff = compute_ela_difference(original_rgb, recompressed)
    stats = compute_statistics(abs_diff, hot)
    fmt = original_format or image.format
    suspicion = compute_suspicion_score(stats, original_format=fmt)
    settings = ElaSettings(
        jpeg_quality=quality,
        amplification_scale=scale,
        hot_pixel_threshold=hot,
    )
    return ElaResult(
        ela_rgb=amplify_for_display(abs_diff, scale),
        ela_heatmap=magnitude_heatmap(abs_diff, scale),
        overlay_image=draw_hotspot_overlay(original_rgb, stats.hotspot_bbox),
        statistics=stats,
        suspicion=suspicion,
        settings=settings,
        source_mode_note=source_mode_note(fmt),
    )


def result_as_dict(result: ElaResult) -> dict[str, Any]:
    stats = asdict(result.statistics)
    return {
        "settings": asdict(result.settings),
        "statistics": stats,
        "suspicion": asdict(result.suspicion),
        "source_mode_note": result.source_mode_note,
        "plain_language": {
            "headline": result.suspicion.simple_headline,
            "status": result.suspicion.simple_status,
            "explanation": result.suspicion.simple_explanation,
        },
    }
