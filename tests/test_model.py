import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from caries_detector import config
from caries_detector.model import (
    Detection,
    InvalidImageError,
    ModelNotFoundError,
    load_model,
    predict,
    prepare_image,
)


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class RecordingModel:
    """Stand-in for a YOLO model: records what predict() is called with."""

    def __init__(self):
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[10.4, 20.6, 110.2, 140.9]]),
            conf=FakeTensor([0.83]),
            xywhn=FakeTensor([[0.1875, 0.25, 0.3125, 0.5]]),
        )
        return [SimpleNamespace(boxes=boxes)]


# --- Guards that don't need the weights ---


def test_predict_passes_pil_rgb_image_on_cpu():
    model = RecordingModel()
    img = Image.new("RGB", (320, 240))
    detections, _ = predict(model, img, conf=0.4)
    source, kwargs = model.calls[0]
    assert isinstance(source, Image.Image) and source.mode == "RGB"
    assert kwargs["device"] == "cpu"
    assert kwargs["conf"] == 0.4
    assert detections == [Detection(confidence=0.83, xyxy=(10, 21, 110, 141), xywhn=(0.1875, 0.25, 0.3125, 0.5))]


def test_predict_rejects_numpy_arrays():
    # Ultralytics reads NumPy arrays as BGR; an RGB array would swap channels silently
    model = RecordingModel()
    with pytest.raises(TypeError):
        predict(model, np.zeros((240, 320, 3), dtype=np.uint8))
    assert model.calls == []


def test_predict_rejects_non_rgb_pil_images():
    with pytest.raises(ValueError):
        predict(RecordingModel(), Image.new("RGBA", (32, 32)))


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "LA"])
def test_prepare_image_converts_to_rgb(mode):
    assert prepare_image(Image.new(mode, (40, 30))).mode == "RGB"


def test_prepare_image_applies_exif_rotation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise to display
    Image.new("RGB", (400, 200)).save(path, exif=exif)
    assert prepare_image(path).size == (200, 400)


def test_prepare_image_keeps_colors():
    img = prepare_image(Image.new("RGB", (8, 8), (255, 0, 0)))
    assert img.getpixel((0, 0)) == (255, 0, 0)


def test_prepare_image_rejects_non_images(tmp_path):
    path = tmp_path / "not_an_image.jpg"
    path.write_bytes(b"this is not a jpeg")
    with pytest.raises(InvalidImageError):
        prepare_image(path)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "missing.pt")


# --- Real model ---

requires_weights = pytest.mark.skipif(not config.MODEL_PATH.is_file(), reason="models/best.pt not found")


@pytest.fixture(scope="module")
def model():
    return load_model(config.MODEL_PATH)


@requires_weights
def test_real_model_is_single_class(model):
    assert model.names == {0: "caries"}


@requires_weights
def test_bad_image_after_loading_model_does_not_autoinstall(model):
    # ultralytics patches Image.open to pip-install pi-heif when an image fails to open
    import os

    assert os.environ.get("YOLO_AUTOINSTALL") == "False"
    with pytest.raises(InvalidImageError):
        prepare_image(io.BytesIO(b"not an image"))


@requires_weights
@pytest.mark.parametrize("size", [(640, 480), (1200, 900)])
def test_real_predict_on_synthetic_image(model, size):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(pixels, "RGB")
    # A very low threshold so the synthetic image is likely to produce some boxes to check
    detections, ms = predict(model, img, conf=0.01)
    assert isinstance(detections, list)
    assert ms > 0
    width, height = size
    for d in detections:
        assert isinstance(d, Detection)
        assert d.class_name == "caries"
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.xyxy
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height
        assert all(0.0 <= v <= 1.0 for v in d.xywhn)
    confidences = [d.confidence for d in detections]
    assert confidences == sorted(confidences, reverse=True)
