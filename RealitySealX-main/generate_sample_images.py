"""
Create synthetic demonstration images for the ELA prototype.

These files are constructed so the UI can show different outcomes:
- a once-saved JPEG that is relatively consistent
- a JPEG with a freshly pasted high-detail region
- a repeatedly recompressed JPEG
- a PNG analysed via JPEG recompression

They are not a labelled forensic dataset.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_images"


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _to_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    out = Image.open(buffer)
    out.load()
    return out.convert("RGB")


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=int(quality), optimize=False)


def build_textured_scene(width: int = 720, height: int = 480, seed: int = 7) -> Image.Image:
    """Photo-like synthetic scene with grain so JPEG has real frequencies to compress."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height)[:, None]
    xx = np.linspace(0, 1, width)[None, :]
    red = 70 + 90 * yy + 25 * np.sin(xx * 9)
    green = 110 + 50 * (1 - yy) + 20 * np.cos(xx * 6)
    blue = 160 - 80 * yy + 15 * np.sin((xx + yy) * 8)
    img = np.stack([red, green, blue], axis=2)
    img = img + rng.normal(0, 10, img.shape)

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([40, 40, 300, 250], fill=(168, 96, 64))
    draw.ellipse([430, 70, 670, 310], fill=(46, 122, 88))
    draw.polygon([(120, 400), (280, 280), (440, 410)], fill=(52, 64, 92))
    draw.text((48, 430), "REALITYSEAL-X demo scene", fill=(20, 20, 20), font=_font(22))
    # Mild blur then grain keeps camera-like texture without looking like a paste.
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.4))
    arr = np.asarray(pil).astype(np.int16) + rng.integers(-6, 7, (height, width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def build_fresh_patch(width: int = 260, height: int = 180, seed: int = 99) -> Image.Image:
    """Uncompressed high-frequency patch — typical of a newly pasted edit."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width))
    checker = ((xx // 5 + yy // 5) % 2) * 200 + 30
    noise = rng.integers(0, 90, (height, width))
    red = np.clip(checker, 0, 255)
    green = np.clip(40 + noise, 0, 255)
    blue = np.clip(255 - checker + noise // 2, 0, 255)
    patch = np.stack([red, green, blue], axis=2).astype(np.uint8)
    image = Image.fromarray(patch, "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, width - 9, height - 9], outline=(255, 255, 0), width=5)
    draw.rectangle([70, 55, 190, 125], fill=(220, 30, 40))
    draw.text((78, 78), "PASTED", fill=(255, 255, 255), font=_font(22))
    return image


def make_unedited(path: Path | None = None) -> Image.Image:
    scene = build_textured_scene()
    jpeg = _to_jpeg(scene, quality=90)
    if path is not None:
        _save_jpeg(jpeg, path, quality=90)
    return jpeg


def make_spliced(path: Path | None = None) -> Image.Image:
    """
    Settle the background as JPEG, paste never-compressed detail, then save.

    That is the classic ELA demonstration: the new region has not yet been
    through the same compression history as the rest of the picture.
    """
    background = _to_jpeg(build_textured_scene(seed=7), quality=70)
    background = _to_jpeg(background, quality=70)
    composed = background.copy()
    composed.paste(build_fresh_patch(), (420, 55))
    jpeg = _to_jpeg(composed, quality=95)
    if path is not None:
        _save_jpeg(jpeg, path, quality=95)
    return jpeg


def make_recompressed(unedited: Image.Image, path: Path | None = None) -> Image.Image:
    once = _to_jpeg(unedited, quality=55)
    twice = _to_jpeg(once, quality=40)
    if path is not None:
        _save_jpeg(twice, path, quality=40)
    return twice


def make_png_and_jpeg_copy(
    path_png: Path | None = None,
    path_jpeg: Path | None = None,
) -> tuple[Image.Image, Image.Image]:
    scene = build_textured_scene(width=640, height=400, seed=21)
    if path_png is not None:
        scene.save(path_png, format="PNG")
    jpeg = _to_jpeg(scene, quality=90)
    if path_jpeg is not None:
        _save_jpeg(jpeg, path_jpeg, quality=90)
    return scene, jpeg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unedited = make_unedited(OUT / "01_unedited.jpg")
    make_spliced(OUT / "02_spliced.jpg")
    make_recompressed(unedited, OUT / "03_recompressed.jpg")
    make_png_and_jpeg_copy(OUT / "04_source.png", OUT / "04_png_converted.jpg")
    print(f"Wrote demonstration images to {OUT}")


if __name__ == "__main__":
    main()
