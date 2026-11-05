"""
Basic metadata and file-integrity helpers.

Missing EXIF is common and is NOT treated as evidence of manipulation.
The SHA-256 digest identifies the exact uploaded bytes; it does not prove
that the image is authentic or unedited.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from PIL import Image, ExifTags


@dataclass
class BasicMetadata:
    filename: str
    format: str | None
    width: int
    height: int
    mode: str
    file_size_bytes: int
    sha256: str
    exif: dict[str, str]
    captured_at_utc: str


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def _stringify_exif_value(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return value.hex()
    if isinstance(value, (list, tuple)):
        return ", ".join(_stringify_exif_value(v) for v in value)
    return str(value)


def extract_exif(image: Image.Image) -> dict[str, str]:
    """Return a string-keyed subset of EXIF tags, or an empty dict if none exist."""
    extracted: dict[str, str] = {}
    try:
        raw = image.getexif()
    except Exception:
        return extracted
    if not raw:
        return extracted

    for tag_id, value in raw.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        # Skip bulky MakerNote / UserComment blobs in the demo UI.
        if name in {"MakerNote", "UserComment", "PrintImageMatching"}:
            continue
        try:
            extracted[name] = _stringify_exif_value(value)
        except Exception:
            extracted[name] = "<unreadable>"
    return extracted


def analyze_file(filename: str, data: bytes, image: Image.Image) -> BasicMetadata:
    return BasicMetadata(
        filename=filename,
        format=image.format,
        width=int(image.size[0]),
        height=int(image.size[1]),
        mode=image.mode,
        file_size_bytes=len(data),
        sha256=sha256_bytes(data),
        exif=extract_exif(image),
        captured_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def metadata_as_dict(meta: BasicMetadata) -> dict[str, Any]:
    return asdict(meta)
