"""Central configuration for the Semiconductor Device Quality Control system.

All paths are relative to the project root so the project runs unchanged on
Windows, macOS and Linux.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------- paths
DATASET_DIR = ROOT / "dataset"
SCENE_DIR = DATASET_DIR / "scenes"          # full conveyor images (+ YOLO detection labels)
DEFECT_DIR = DATASET_DIR / "defects"        # device crops (+ YOLO-seg polygons + U-Net masks)
ANNOTATION_DIR = DATASET_DIR / "annotations"  # ground truth per scene (JSON)
SAMPLE_DIR = DATASET_DIR / "samples"        # a few demo images committed to git

MODEL_DIR = ROOT / "models"
DETECTOR_WEIGHTS = MODEL_DIR / "detector_yolo11n.pt"
YOLO_SEG_WEIGHTS = MODEL_DIR / "segmenter_yolo11n_seg.pt"
UNET_WEIGHTS = MODEL_DIR / "segmenter_unet.pt"

RESULT_DIR = ROOT / "results"
GRAPH_DIR = RESULT_DIR / "graphs"
REPORT_DIR = RESULT_DIR / "reports"
PRED_DIR = RESULT_DIR / "predictions"
RUNS_DIR = RESULT_DIR / "runs"              # ultralytics training runs
INSPECTION_IMG_DIR = RESULT_DIR / "inspections"
DB_PATH = RESULT_DIR / "inspections.db"
BEST_PIPELINE_FILE = REPORT_DIR / "best_pipeline.json"
THRESHOLDS_FILE = REPORT_DIR / "tuned_thresholds.json"

RULES_FILE = ROOT / "configs" / "quality_rules.yaml"

OUTPUT_DIRS = [MODEL_DIR, GRAPH_DIR, REPORT_DIR, PRED_DIR, RUNS_DIR, INSPECTION_IMG_DIR]

# ----------------------------------------------------------------- classes
DEVICE_CLASS = "ic_device"
DEFECT_CLASSES = ["crack", "scratch", "contamination", "chip", "missing_pin"]
DEFECT_TO_ID = {c: i for i, c in enumerate(DEFECT_CLASSES)}
DECISIONS = ["PASS", "REWORK", "SCRAP"]           # ordered by severity

# ---------------------------------------------------------------- camera
SCENE_SIZE = 512          # rendered camera frame (pixels)
PX_PER_MM = 12.0          # camera calibration: pixels per millimetre on the conveyor plane

# --------------------------------------------------------------- dataset
SEED = 42
N_SCENES = {"train": 500, "val": 100, "test": 150}
CROP_SIZE = 256           # device crops fed to the segmentation models
CROP_PAD = 0.08           # padding around the detected device box

# --------------------------------------------------------------- training
DETECTOR = {"base": "yolo11n.pt", "imgsz": 416, "epochs": 10, "batch": 16, "patience": 5}
YOLO_SEG = {"base": "yolo11n-seg.pt", "imgsz": 256, "epochs": 25, "batch": 16, "patience": 8}
UNET = {"base_channels": 16, "epochs": 30, "batch": 16, "lr": 2e-3, "patience": 8, "patch": 128}

# -------------------------------------------------------------- inference
DET_CONF = 0.40
SEG_CONF = 0.25           # defaults; replaced by values tuned on the validation split
MIN_DEFECT_PIXELS = 6     # ignore specks smaller than this (crop pixels)
TUNE_SEG_CONF = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TUNE_MIN_PIXELS = [6, 15, 30, 60, 100, 150, 250]

# GitHub release with the pretrained YOLO11 weights (used when not cached locally)
YOLO_ASSET_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"


def ensure_dirs() -> None:
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def load_rules(path: Path = RULES_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
