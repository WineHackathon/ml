from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_CATALOG_PIXELS = 64_000_000


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    if not data:
        raise InvalidImage("empty image")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage(f"image exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > max_pixels:
                raise InvalidImage(f"image exceeds {max_pixels} pixels")
            source.load()
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage("unsupported or corrupt image") from exc


def retrieval_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    """Whole image plus a conservative central label/bottle crop."""
    width, height = image.size
    left, top = int(width * 0.15), int(height * 0.15)
    right, bottom = max(left + 1, int(width * 0.85)), max(top + 1, int(height * 0.85))
    return [("whole", image), ("center70", image.crop((left, top, right, bottom)))]
