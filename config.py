"""
REALITYSEAL-X prototype configuration.

Thresholds and scoring weights are HEURISTICS for a local demonstration.
They are NOT scientifically validated and MUST be recalibrated on a labelled
dataset before any research or forensic claim is made.
"""

from pathlib import Path

PROJECT_NAME = "REALITYSEAL-X"
PROJECT_SUBTITLE = "Image Integrity and ELA-Based Tampering Analysis — Prototype"
PROJECT_LAYER = "Layer 1: Error Level Analysis (ELA) prototype"

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGES_DIR = ROOT_DIR / "sample_images"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# --- Upload limits ----------------------------------------------------------
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_HINTS = {"image/jpeg", "image/png", "image/jpg"}

# --- ELA recompression ------------------------------------------------------
JPEG_RECOMPRESS_QUALITY = 90
ELA_AMPLIFICATION_SCALE = 20.0
HOT_PIXEL_THRESHOLD = 8.0
BLOCK_SIZE = 16
BLOCK_OUTLIER_K = 3.5
BLOCK_OUTLIER_RATIO = 2.2
BLOCK_OUTLIER_MIN_DIFFERENCE = 1.5
BLOCK_CONTRAST_PERCENTILE = 95.0
BLOCK_CONTRAST_MIN_TYPICAL = 1.0
HOTSPOT_MIN_DIFFERENCE = 3.0

# --- Suspicion score caps (need dataset calibration) ------------------------
# Global mean is a weak splice signal (a small paste barely changes the average).
# Local block contrast and outlier blocks are weighted more so a pasted region
# can raise the score even when most of the image looks normal.
SCORE_MEAN_CAP = 6.0
SCORE_STD_CAP = 8.0
SCORE_MAX_CAP = 60.0
SCORE_HOT_PCT_CAP = 12.0
SCORE_P99_CAP = 18.0
SCORE_CONTRAST_CAP = 3.5
SCORE_OUTLIER_BLOCK_CAP = 12.0

WEIGHT_MEAN = 0.08
WEIGHT_STD = 0.10
WEIGHT_MAX = 0.07
WEIGHT_HOT = 0.10
WEIGHT_P99 = 0.18
WEIGHT_CONTRAST = 0.27
WEIGHT_OUTLIER_BLOCKS = 0.20

LOW_SCORE_MAX = 24
MODERATE_SCORE_MAX = 62

SCORE_WARNING = (
    "This is a heuristic ELA score, not a calibrated probability that "
    "the image is manipulated."
)

# Technical interpretations (kept for the detailed panel and report)
INTERPRETATION_HIGH = (
    "Elevated ELA differences were observed. These may be associated with "
    "editing, recompression, or other image-processing operations. "
    "Further verification is required."
)
INTERPRETATION_LOW = (
    "This test did not find a strong ELA pattern that stands out from nearby "
    "areas. That does not confirm that the image is unchanged."
)
INTERPRETATION_MODERATE = (
    "Moderate ELA differences were observed. This can occur with editing, "
    "recompression, camera processing, or natural variation in texture. "
    "Further verification is required."
)

# Plain-language copy for non-technical readers
SIMPLE_HEADLINE_LOW = "No clear sign of editing found"
SIMPLE_STATUS_LOW = "This test found no strong unusual area"
SIMPLE_BODY_LOW = (
    "In everyday terms: this JPEG check did not find a clear patch that stands "
    "out from the rest of the picture. The image may be unchanged, but this "
    "test can miss edits, so this result is not proof either way."
)

SIMPLE_HEADLINE_MODERATE = "Some parts look different from the rest"
SIMPLE_STATUS_MODERATE = "Worth checking more closely"
SIMPLE_BODY_MODERATE = (
    "In everyday terms: some parts changed differently when the picture was "
    "re-saved as JPEG. Editing can cause this, but so can messaging apps, "
    "social-media compression, or a camera. This is a reason to look closer, "
    "not a yes-or-no answer."
)

SIMPLE_HEADLINE_HIGH = "A strong unusual pattern was found"
SIMPLE_STATUS_HIGH = "Possible editing or unusual image processing"
SIMPLE_BODY_HIGH = (
    "In everyday terms: one or more areas changed much more than their "
    "surroundings when re-saved as JPEG. This can happen when a region was "
    "edited, but other image processing can cause it too. Check the highlighted "
    "area and verify the image another way before drawing a conclusion."
)

HEATMAP_HINT = (
    "How to read the heatmap: darker areas are more consistent under this test. "
    "Warmer or brighter areas changed more when the picture was re-saved as JPEG. "
    "If a small region is much brighter than its surroundings, that is the area "
    "to inspect."
)

LIMITATIONS = [
    "ELA is not a reliable standalone detector of all image manipulation.",
    "Recompression and social-media processing can affect ELA results.",
    "Copy-move edits may not be detected reliably.",
    "Different image regions can naturally have different compression characteristics.",
    "A high ELA score is not proof of manipulation.",
    "A low ELA score is not proof of authenticity.",
    "The ELA score describes the strength of a pattern in this test; it is not "
    "the percentage chance that an image was edited.",
    "This prototype does not verify the complete origin or history of an image.",
]
