"""Central configuration for the Semiconductor Wafer AI project.

All paths are relative to the project root, so the project runs unchanged on
Windows, macOS and Linux.
"""
from pathlib import Path

# --------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
LABEL_FILE = DATASET_DIR / "labels.csv"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
GRAPH_DIR = RESULT_DIR / "graphs"
CM_DIR = RESULT_DIR / "confusion_matrices"
PREDICTION_DIR = RESULT_DIR / "predictions"
REPORT_DIR = RESULT_DIR / "reports"
REPORT_DOC_DIR = BASE_DIR / "report"

SPLIT_FILE = REPORT_DIR / "data_split.csv"
BEST_MODEL_FILE = REPORT_DIR / "best_model.json"

OUTPUT_DIRS = [MODEL_DIR, GRAPH_DIR, CM_DIR, PREDICTION_DIR, REPORT_DIR]

# ------------------------------------------------------------------- classes
CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-Full", "Random", "Scratch", "None",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Wafer-map encoding used by the synthetic generator (same idea as WM-811K):
# pixel 0 = outside the wafer, 127 = good die, 255 = defective die
PIXEL_OUTSIDE, PIXEL_GOOD, PIXEL_DEFECT = 0, 127, 255

# ------------------------------------------------------------------- dataset
SYNTHETIC_PER_CLASS = 100          # 9 x 100 = 900 images
IMG_SIZE = 64
SEED = 42

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# ------------------------------------------------------------------ training
# Each model: input size used inside the network, epochs, batch size, learning rate.
MODELS = {
    "CNN": {
        "file": "cnn_model.keras",
        "slug": "cnn",
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-3,
    },
    "ResNet50": {
        "file": "resnet50_model.keras",
        "slug": "resnet50",
        "input_size": 224,       # 64x64 grayscale is resized + converted to 224x224x3 inside the model
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
    },
    "EfficientNetB0": {
        "file": "efficientnetb0_model.keras",
        "slug": "efficientnetb0",
        "input_size": 224,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
    },
}

EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
USE_AUGMENTATION = True            # small rotations/flips on the training set only

# ----------------------------------------------------------------- selection
# Best model = highest macro F1; ties (within this margin) broken by accuracy, then speed.
TIE_MARGIN = 0.005

# ----------------------------------------------------------------- prediction
CONFIDENCE_LEVELS = [            # (minimum probability, label)
    (0.90, "Very high confidence"),
    (0.70, "High confidence"),
    (0.50, "Moderate confidence"),
    (0.00, "Low confidence"),
]
