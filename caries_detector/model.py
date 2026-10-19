"""Model loading, image preprocessing and inference. No UI code here."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Union

from PIL import Image, ImageOps

from . import config


class ModelNotFoundError(FileNotFoundError):
    """Raised when the weights file does not exist."""


class InvalidImageError(ValueError):
    """Raised when an input cannot be read as an image."""


@dataclass(frozen=True)
class Detection:
    confidence: float
    xyxy: tuple[int, int, int, int]  # pixel corners: x1, y1, x2, y2
    xywhn: tuple[float, float, float, float]  # normalized center x, center y, width, height
    class_name: str = config.CLASS_NAME

    @property
    def width(self) -> int:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> int:
        return self.xyxy[3] - self.xyxy[1]

    def to_dict(self) -> dict:
        return asdict(self)


def load_model(path: Path = config.MODEL_PATH):
    """Load the YOLO weights at `path`."""
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(f"Model weights not found at {path}")
    # Importing ultralytics patches PIL.Image.open to pip-install pi-heif whenever an
    # image fails to open (e.g. a corrupt upload). Turn auto-install off before importing.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    # Imported here so the rest of the package works without importing torch
    from ultralytics import YOLO

    return YOLO(str(path))


def warm_up(model) -> None:
    """Run one prediction on a blank image so the first real one isn't slowed by setup (~3 s on CPU)."""
    blank = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))
    model.predict(blank, imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False)


def prepare_image(source: Union[str, Path, IO[bytes], Image.Image]) -> Image.Image:
    """Open an image, apply its EXIF rotation and convert it to RGB."""
    try:
        img = source if isinstance(source, Image.Image) else Image.open(source)
        img.load()
    except Exception as exc:
        # The underlying error is kept as __cause__; it isn't useful to show users
        raise InvalidImageError("Could not read the file as an image. Is it a valid .jpg or .png?") from exc
    # Phone photos often store their rotation in EXIF instead of in the pixels
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            # Composite transparency onto white instead of black
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")
    return img


def predict(model, pil_image: Image.Image, conf: float = config.DEFAULT_CONF) -> tuple[list[Detection], float]:
    """Run the detector on an RGB PIL image.

    Returns the detections (highest confidence first) and the inference time in ms.
    """
    # Ultralytics treats NumPy arrays as BGR, so an RGB array would silently swap
    # the red and blue channels. Only accept PIL images, which it reads as RGB.
    if not isinstance(pil_image, Image.Image):
        raise TypeError(f"predict() expects a PIL image, got {type(pil_image).__name__}")
    if pil_image.mode != "RGB":
        raise ValueError(f"predict() expects an RGB image, got mode {pil_image.mode!r}; use prepare_image()")

    start = time.perf_counter()
    result = model.predict(pil_image, conf=conf, imgsz=config.IMG_SIZE, device=config.DEVICE, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    width, height = pil_image.size
    boxes = result.boxes
    detections = []
    for xyxy, score, xywhn in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()):
        x1, y1, x2, y2 = (round(v) for v in xyxy)
        detections.append(
            Detection(
                confidence=float(score),
                xyxy=(
                    min(max(x1, 0), width),
                    min(max(y1, 0), height),
                    min(max(x2, 0), width),
                    min(max(y2, 0), height),
                ),
                xywhn=tuple(float(v) for v in xywhn),
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections, elapsed_ms
