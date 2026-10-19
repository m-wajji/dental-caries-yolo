"""Paths, defaults, theme colors and model metadata used across the app."""

from pathlib import Path

# Paths (all relative to the repository root, independent of the working directory)
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best.pt"
OUTPUTS_DIR = ROOT_DIR / "assets" / "outputs"
SAMPLE_IMAGES_DIR = ROOT_DIR / "data" / "yolo" / "test" / "images"

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")

# Inference
DEVICE = "cpu"
IMG_SIZE = 640
DEFAULT_CONF = 0.4
CONF_MIN = 0.05
CONF_MAX = 0.95
CONF_STEP = 0.05
CLASS_NAME = "caries"

# Theme (keep in sync with .streamlit/config.toml)
PRIMARY_COLOR = "#15803d"  # green-700
BOX_COLOR = "#16a34a"  # green-600
LABEL_TEXT_COLOR = "#ffffff"
LIGHT_GREEN = "#f0fdf4"  # green-50
TEXT_COLOR = "#14532d"  # green-900

# Model metadata, shown in the "Model info" panel
MODEL_INFO = {
    "Architecture": "YOLO26n (Ultralytics 8.4.x), fine-tuned from COCO weights",
    "Input size": f"{IMG_SIZE} px",
    "Classes": "1 (caries; permanent and primary tooth classes merged)",
    "Training data": (
        "2,200+ dentist-annotated intraoral phone photos "
        "(Aga Khan University dataset, Mithi, Sindh, Pakistan; CC BY 4.0)"
    ),
}

# Held-out test set: 285 images of people not seen during training
TEST_METRICS = {
    "Precision": 0.887,
    "Recall": 0.885,
    "mAP50": 0.942,
    "mAP50-95": 0.744,
}

# Precision and recall on the test set at different confidence thresholds
THRESHOLD_TABLE = [
    {"conf": 0.2, "precision": 0.782, "recall": 0.956},
    {"conf": 0.3, "precision": 0.815, "recall": 0.939},
    {"conf": 0.4, "precision": 0.840, "recall": 0.928},
    {"conf": 0.5, "precision": 0.855, "recall": 0.905},
    {"conf": 0.6, "precision": 0.885, "recall": 0.885},
    {"conf": 0.8, "precision": 0.915, "recall": 0.753},
]

DISCLAIMER = "Learning project, not a medical device. Results must be confirmed by a dentist."
