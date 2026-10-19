import json
from datetime import datetime

from PIL import Image

from caries_detector.model import Detection
from caries_detector.storage import safe_stem, save_result

REQUIRED_KEYS = {
    "original_filename",
    "timestamp",
    "model_path",
    "confidence_threshold",
    "image_width",
    "image_height",
    "inference_ms",
    "detections",
}

TIMESTAMP = datetime(2026, 10, 19, 14, 30, 5)


def sample_detections():
    return [
        Detection(confidence=0.91, xyxy=(10, 20, 110, 140), xywhn=(0.1875, 0.25, 0.3125, 0.5)),
        Detection(confidence=0.45, xyxy=(200, 100, 260, 180), xywhn=(0.7187, 0.5833, 0.1875, 0.3333)),
    ]


def save(tmp_path, name="my photo (1).jpg", detections=None):
    img = Image.new("RGB", (320, 240), "white")
    return save_result(
        img,
        original_filename=name,
        detections=sample_detections() if detections is None else detections,
        conf=0.4,
        inference_ms=97.3,
        out_dir=tmp_path / "outputs",
        timestamp=TIMESTAMP,
    )


def test_creates_both_files_and_folder(tmp_path):
    image_path, json_path = save(tmp_path)
    assert image_path.is_file() and json_path.is_file()
    assert image_path.name == "my_photo_1_20261019-143005_annotated.jpg"
    assert json_path.name == "my_photo_1_20261019-143005.json"
    with Image.open(image_path) as img:
        assert img.size == (320, 240)


def test_json_contains_required_keys(tmp_path):
    _, json_path = save(tmp_path)
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert REQUIRED_KEYS <= record.keys()
    assert record["original_filename"] == "my photo (1).jpg"
    assert record["confidence_threshold"] == 0.4
    assert (record["image_width"], record["image_height"]) == (320, 240)
    assert len(record["detections"]) == 2
    first = record["detections"][0]
    assert {"confidence", "xyxy", "xywhn"} <= first.keys()
    assert first["xyxy"] == [10, 20, 110, 140]


def test_empty_detections_are_saved(tmp_path):
    _, json_path = save(tmp_path, detections=[])
    assert json.loads(json_path.read_text(encoding="utf-8"))["detections"] == []


def test_same_name_twice_does_not_overwrite(tmp_path):
    first = save(tmp_path)
    first_bytes = first[1].read_bytes()
    second = save(tmp_path)
    third = save(tmp_path)
    paths = {*first, *second, *third}
    assert len(paths) == 6
    assert all(p.is_file() for p in paths)
    assert first[1].read_bytes() == first_bytes
    assert second[0].name == "my_photo_1_20261019-143005_1_annotated.jpg"


def test_safe_stem():
    assert safe_stem("my photo (1).jpg") == "my_photo_1"
    assert safe_stem("../../etc/passwd") == "passwd"
    assert safe_stem("Ünïcode name!.png") == "n_code_name"
    assert safe_stem("???.jpg") == "image"
    assert len(safe_stem("a" * 300 + ".jpg")) <= 60
