"""Safe loading and validation of uploaded images. Original files are never overwritten."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import config


class ImageLoadError(ValueError):
    """Raised when an upload cannot be used for analysis."""


def validate_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise ImageLoadError(f"Unsupported file type '{suffix}'. Allowed types: {allowed}.")
    return suffix


def validate_size(data: bytes) -> None:
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ImageLoadError(
            f"File is {len(data)} bytes, which exceeds the {config.MAX_UPLOAD_MB} MB limit."
        )
    if len(data) == 0:
        raise ImageLoadError("The uploaded file is empty.")


def load_image_from_bytes(data: bytes, filename: str) -> Image.Image:
    """
    Validate type/size and load pixels into memory.

    Pillow's verify() can leave the file pointer in an unusable state, so the
    image is opened twice: once to verify structure, then again to load pixels.
    """
    validate_filename(filename)
    validate_size(data)

    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
    except UnidentifiedImageError as exc:
        raise ImageLoadError("The file could not be identified as a valid image.") from exc
    except Exception as exc:
        raise ImageLoadError(f"The image appears corrupted or unreadable: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ImageLoadError(f"The image could not be fully decoded: {exc}") from exc

    if image.size[0] < 8 or image.size[1] < 8:
        raise ImageLoadError("Image is too small to analyse (minimum 8×8 pixels).")
    return image
