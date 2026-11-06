"""Build downloadable text and JSON analysis reports. No original pixel payload is stored."""

from __future__ import annotations

import json
from typing import Any

import config
from ela_analyzer import ElaResult, result_as_dict
from metadata_analyzer import BasicMetadata, metadata_as_dict


def build_report_payload(meta: BasicMetadata, ela: ElaResult) -> dict[str, Any]:
    is_png = (meta.format or "").upper() == "PNG"
    return {
        "project_name": config.PROJECT_NAME,
        "layer": config.PROJECT_LAYER,
        "analysis_datetime_utc": meta.captured_at_utc,
        "uploaded_filename": meta.filename,
        "image": {
            "format": meta.format,
            "width": meta.width,
            "height": meta.height,
            "mode": meta.mode,
            "file_size_bytes": meta.file_size_bytes,
        },
        "sha256": meta.sha256,
        "sha256_note": (
            "SHA-256 identifies the exact uploaded file bytes and can show whether "
            "that file later changes. It does not establish authenticity."
        ),
        "exif_present": bool(meta.exif),
        "exif": meta.exif,
        "exif_note": (
            "Missing EXIF metadata is common (exports, screenshots, messaging apps) "
            "and is not treated as proof of manipulation."
        ),
        "ela": result_as_dict(ela),
        "heuristic_suspicion_score": ela.suspicion.score,
        "ela_contrast_score": ela.suspicion.score,
        "ela_score_label": (
            "JPEG conversion difference" if is_png else "ELA contrast score"
        ),
        "indicator_category": ela.suspicion.indicator,
        "flagged": (
            not is_png
            and ela.suspicion.score > config.LOW_SCORE_MAX
        ),
        "plain_language_headline": ela.suspicion.simple_headline,
        "plain_language_status": ela.suspicion.simple_status,
        "plain_language_explanation": ela.suspicion.simple_explanation,
        "interpretation": ela.suspicion.interpretation,
        "score_warning": config.SCORE_WARNING,
        "limitations": config.LIMITATIONS,
        "disclaimer": (
            "This report is produced by a college research prototype implementing "
            "Error Level Analysis only. It is not a complete authenticity, provenance, "
            "or legal-forensic determination."
        ),
    }


def report_json(meta: BasicMetadata, ela: ElaResult) -> str:
    return json.dumps(build_report_payload(meta, ela), indent=2)


def report_text(meta: BasicMetadata, ela: ElaResult) -> str:
    payload = build_report_payload(meta, ela)
    stats = payload["ela"]["statistics"]
    settings = payload["ela"]["settings"]
    suspicion = payload["ela"]["suspicion"]
    lines = [
        f"{payload['project_name']}",
        payload["layer"],
        "=" * 72,
        f"Analysis datetime (UTC): {payload['analysis_datetime_utc']}",
        f"Uploaded filename: {payload['uploaded_filename']}",
        "",
        "Image",
        "-----",
        f"Format: {payload['image']['format']}",
        f"Dimensions: {payload['image']['width']} x {payload['image']['height']}",
        f"Mode: {payload['image']['mode']}",
        f"File size (bytes): {payload['image']['file_size_bytes']}",
        "",
        "Basic metadata and file integrity",
        "---------------------------------",
        f"SHA-256: {payload['sha256']}",
        payload["sha256_note"],
        f"EXIF tags present: {payload['exif_present']}",
        payload["exif_note"],
    ]
    if payload["exif"]:
        lines.append("EXIF (available tags):")
        for key, value in payload["exif"].items():
            lines.append(f"  - {key}: {value}")
    else:
        lines.append("EXIF: none readable in this file.")

    lines.extend(
        [
            "",
            "ELA settings",
            "------------",
            f"JPEG recompress quality: {settings['jpeg_quality']}",
            f"Amplification scale: {settings['amplification_scale']}",
            f"Hot-pixel threshold (raw difference): {settings['hot_pixel_threshold']}",
            "",
            "ELA statistics",
            "--------------",
            f"Mean ELA difference: {stats['mean_difference']:.4f}",
            f"Maximum ELA difference: {stats['max_difference']:.4f}",
            f"Standard deviation: {stats['std_difference']:.4f}",
            (
                "Percentage of pixels above threshold: "
                f"{stats['percent_pixels_above_threshold']:.4f}%"
            ),
            f"99th percentile ELA: {stats.get('p99_difference', 0):.4f}",
            f"Block contrast ratio: {stats.get('block_contrast_ratio', 1):.4f}",
            f"Outlier block percent: {stats.get('outlier_block_percent', 0):.4f}",
            f"Pixel count: {stats['pixel_count']}",
            payload["ela"]["source_mode_note"],
            "",
            "ELA contrast score and result",
            "-----------------------------",
            f"{payload['ela_score_label']}: {payload['ela_contrast_score']}/100",
            f"Result: {payload['indicator_category']}",
            f"Plain-language headline: {payload['plain_language_headline']}",
            f"Simple reading: {payload['plain_language_status']}",
            payload["plain_language_explanation"],
            f"Technical interpretation: {payload['interpretation']}",
            payload["score_warning"],
            suspicion["formula_note"],
            "Score components:",
        ]
    )
    for key, value in suspicion["components"].items():
        lines.append(f"  - {key}: {value}")

    lines.extend(["", "Limitations", "-----------"])
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    lines.extend(["", payload["disclaimer"], ""])
    return "\n".join(lines)
