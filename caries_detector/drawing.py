"""Draw detections on an image in the app's theme."""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from . import config
from .model import Detection


def _style_for(size: tuple[int, int]) -> tuple[int, int]:
    """Line width and font size scaled to the image (photos range from ~700 to 3,000+ px)."""
    short_side = min(size)
    line_width = max(2, round(short_side / 300))
    font_size = max(12, round(short_side / 45))
    return line_width, font_size


def draw_detections(pil_image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    """Return a copy of `pil_image` with a box and a `caries 0.83` tag for each detection."""
    annotated = pil_image.convert("RGB")  # always a copy
    if not detections:
        return annotated

    width, height = annotated.size
    line_width, font_size = _style_for(annotated.size)
    font = ImageFont.load_default(size=font_size)
    pad = max(2, font_size // 4)
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=config.BOX_COLOR, width=line_width)

        label = f"{det.class_name} {det.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tag_w = right - left + 2 * pad
        tag_h = bottom - top + 2 * pad

        # Put the tag above the box; if there's no room, put it inside the top edge.
        tag_x = x1
        tag_y = y1 - tag_h if y1 - tag_h >= 0 else y1
        # Keep the whole tag on the image
        tag_x = min(max(tag_x, 0), max(width - tag_w, 0))
        tag_y = min(max(tag_y, 0), max(height - tag_h, 0))

        draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h), fill=config.BOX_COLOR)
        draw.text((tag_x + pad - left, tag_y + pad - top), label, font=font, fill=config.LABEL_TEXT_COLOR)

    return annotated
