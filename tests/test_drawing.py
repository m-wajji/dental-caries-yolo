import pytest
from PIL import Image, ImageChops

from caries_detector import config
from caries_detector.drawing import draw_detections
from caries_detector.model import Detection


def make_detection(x1, y1, x2, y2, conf=0.83, size=(800, 600)):
    w, h = size
    xywhn = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
    return Detection(confidence=conf, xyxy=(x1, y1, x2, y2), xywhn=xywhn)


@pytest.mark.parametrize("size", [(700, 500), (2400, 1800)])
def test_annotated_image_has_same_size(size):
    img = Image.new("RGB", size, "white")
    out = draw_detections(img, [make_detection(100, 100, 300, 300, size=size)])
    assert out.size == size
    assert out.mode == "RGB"


def test_boxes_use_theme_color():
    img = Image.new("RGB", (800, 600), "white")
    out = draw_detections(img, [make_detection(100, 200, 400, 500)])
    box_rgb = Image.new("RGB", (1, 1), config.BOX_COLOR).getpixel((0, 0))
    # Left edge of the box, below the label tag
    assert out.getpixel((100, 400)) == box_rgb


def test_zero_detections_returns_unchanged_copy():
    img = Image.new("RGB", (640, 480), (10, 120, 200))
    out = draw_detections(img, [])
    assert out is not img
    assert ImageChops.difference(out, img).getbbox() is None


def test_input_image_is_not_modified():
    img = Image.new("RGB", (640, 480), "white")
    before = img.copy()
    draw_detections(img, [make_detection(50, 50, 200, 200)])
    assert ImageChops.difference(img, before).getbbox() is None


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 50, 40),  # top-left corner: no room above for the tag
        (760, 0, 800, 30),  # top-right corner: tag would run off the right edge
        (0, 570, 40, 600),  # bottom-left corner
        (780, 580, 800, 600),  # tiny box in the bottom-right corner
        (0, 0, 800, 600),  # box covering the whole image
    ],
)
def test_labels_near_edges_do_not_raise(box):
    img = Image.new("RGB", (800, 600), "white")
    out = draw_detections(img, [make_detection(*box, conf=0.99)])
    assert out.size == (800, 600)


def test_tiny_image_does_not_raise():
    # Label tag wider than the image
    img = Image.new("RGB", (30, 20), "white")
    out = draw_detections(img, [make_detection(5, 5, 25, 15, size=(30, 20))])
    assert out.size == (30, 20)
