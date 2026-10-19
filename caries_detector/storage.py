"""Save annotated images and detection JSON to disk."""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from . import config
from .model import Detection

JPEG_QUALITY = 92


def safe_stem(filename: str, max_length: int = 60) -> str:
    """Turn a filename into a safe stem: letters, digits, '-' and '_' only."""
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")
    return stem[:max_length] or "image"


def build_record(
    original_filename: str,
    timestamp: datetime,
    model_path: Path,
    conf: float,
    image_size: tuple[int, int],
    inference_ms: float,
    detections: Sequence[Detection],
) -> dict:
    """The JSON content saved next to each annotated image."""
    width, height = image_size
    model_path = Path(model_path)
    # Store the path relative to the repo so the JSON doesn't leak local folder names
    if model_path.is_absolute() and model_path.is_relative_to(config.ROOT_DIR):
        model_path = model_path.relative_to(config.ROOT_DIR)
    return {
        "original_filename": original_filename,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "model_path": model_path.as_posix(),
        "confidence_threshold": round(conf, 4),
        "image_width": width,
        "image_height": height,
        "inference_ms": round(inference_ms, 1),
        "num_detections": len(detections),
        "detections": [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "xyxy": list(d.xyxy),
                "xywhn": [round(v, 6) for v in d.xywhn],
            }
            for d in detections
        ],
    }


def record_to_json(record: dict) -> str:
    return json.dumps(record, indent=2)


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def _unique_base(out_dir: Path, base: str) -> str:
    """Return `base`, or `base_1`, `base_2`, ... so neither output file already exists."""
    candidate, n = base, 0
    while (out_dir / f"{candidate}_annotated.jpg").exists() or (out_dir / f"{candidate}.json").exists():
        n += 1
        candidate = f"{base}_{n}"
    return candidate


def save_result(
    annotated: Image.Image,
    original_filename: str,
    detections: Sequence[Detection],
    conf: float,
    inference_ms: float,
    image_size: Optional[tuple[int, int]] = None,
    model_path: Path = config.MODEL_PATH,
    out_dir: Path = config.OUTPUTS_DIR,
    timestamp: Optional[datetime] = None,
) -> tuple[Path, Path]:
    """Write `<stem>_<YYYYmmdd-HHMMSS>_annotated.jpg` and `<stem>_<YYYYmmdd-HHMMSS>.json`.

    Never overwrites: on a name clash a `_1`, `_2`, ... suffix is added.
    Returns (image_path, json_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now()

    base = _unique_base(out_dir, f"{safe_stem(original_filename)}_{timestamp:%Y%m%d-%H%M%S}")
    image_path = out_dir / f"{base}_annotated.jpg"
    json_path = out_dir / f"{base}.json"

    record = build_record(
        original_filename=original_filename,
        timestamp=timestamp,
        model_path=model_path,
        conf=conf,
        image_size=image_size or annotated.size,
        inference_ms=inference_ms,
        detections=detections,
    )
    # "x" mode fails instead of overwriting if a file appears in the meantime
    with open(image_path, "xb") as f:
        f.write(image_to_jpeg_bytes(annotated))
    with open(json_path, "x", encoding="utf-8") as f:
        f.write(record_to_json(record))
    return image_path, json_path
