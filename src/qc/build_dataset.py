"""Generate the synthetic dataset in all the formats the models need.

    dataset/scenes/<split>/images/*.png     camera frames
    dataset/scenes/<split>/labels/*.txt     YOLO detection labels (class 0 = ic_device)
    dataset/scenes/data.yaml                YOLO detection dataset file
    dataset/defects/<split>/images/*.png    256x256 device crops
    dataset/defects/<split>/labels/*.txt    YOLO segmentation polygons (5 defect classes)
    dataset/defects/<split>/masks/*.png     semantic masks for U-Net (0 = background, 1..5)
    dataset/defects/data.yaml               YOLO segmentation dataset file
    dataset/annotations/<split>.json        ground truth: boxes, defects, measurements, decisions

Splits are made at SCENE level (a device crop always belongs to the same split
as its camera frame) so there is no leakage between train, val and test.

Usage:
    python run_project.py generate
"""
from __future__ import annotations

import json
import shutil
import sys

import cv2
import numpy as np
import yaml

from qc import config as C
from qc.geometry import crop_mask, square_crop
from qc.synth import device_ground_truth, render_scene

SPLIT_SEEDS = {"train": 1000, "val": 2000, "test": 3000}


def _yolo_box(box, W, H):
    x1, y1, x2, y2 = box
    return f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def _polygons(mask: np.ndarray, size: int):
    """Mask -> list of normalised polygons (one per connected component)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 1 and len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:  # 1-2 pixel blobs: use their tiny bounding square
            x, y = pts.min(0)
            pts = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], np.float32)
        out.append((pts / size).clip(0, 1))
    return out


def build_split(split: str, n_scenes: int, rules: dict) -> dict:
    rng = np.random.default_rng(SPLIT_SEEDS[split] + C.SEED)
    sdir, ddir = C.SCENE_DIR / split, C.DEFECT_DIR / split
    for sub in ("images", "labels"):
        (sdir / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels", "masks"):
        (ddir / sub).mkdir(parents=True, exist_ok=True)

    annotations = []
    n_crops = 0
    for i in range(n_scenes):
        scene = render_scene(rng)
        name = f"{split}_{i:04d}"
        H, W = scene.image.shape[:2]
        cv2.imwrite(str(sdir / "images" / f"{name}.png"), scene.image)
        (sdir / "labels" / f"{name}.txt").write_text(
            "\n".join(_yolo_box(d.box, W, H) for d in scene.devices) + "\n")

        devices = []
        for k, dev in enumerate(scene.devices):
            gt = device_ground_truth(dev, rules)
            # training crops get a small random shift/scale to mimic detector box errors
            jitter = rng.uniform([-0.04, -0.04, -0.08], [0.04, 0.04, 0.08]) if split == "train" else None
            crop, info = square_crop(scene.image, dev.box, jitter=jitter)
            cname = f"{name}_d{k}"
            cv2.imwrite(str(ddir / "images" / f"{cname}.png"), crop)
            semantic = np.zeros((C.CROP_SIZE, C.CROP_SIZE), np.uint8)
            lines = []
            for d in dev.defects:
                m = crop_mask(d.mask, info)
                if not m.any():
                    continue
                semantic[m] = C.DEFECT_TO_ID[d.cls] + 1
                for poly in _polygons(m, C.CROP_SIZE):
                    lines.append(f"{C.DEFECT_TO_ID[d.cls]} " + " ".join(f"{v:.5f}" for v in poly.ravel()))
            (ddir / "labels" / f"{cname}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            cv2.imwrite(str(ddir / "masks" / f"{cname}.png"), semantic)
            gt["crop"] = cname
            devices.append(gt)
            n_crops += 1
        annotations.append({"image": f"scenes/{split}/images/{name}.png", "devices": devices})

    C.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    (C.ANNOTATION_DIR / f"{split}.json").write_text(json.dumps(
        {"split": split, "px_per_mm": C.PX_PER_MM, "scene_size": C.SCENE_SIZE,
         "synthetic": True, "scenes": annotations}, indent=1))
    return {"scenes": n_scenes, "devices": n_crops}


def write_yaml_files() -> None:
    det = {"path": str(C.SCENE_DIR.resolve()), "train": "train/images", "val": "val/images",
           "test": "test/images", "names": {0: C.DEVICE_CLASS}}
    seg = {"path": str(C.DEFECT_DIR.resolve()), "train": "train/images", "val": "val/images",
           "test": "test/images", "names": dict(enumerate(C.DEFECT_CLASSES))}
    (C.SCENE_DIR / "data.yaml").write_text(yaml.safe_dump(det, sort_keys=False))
    (C.DEFECT_DIR / "data.yaml").write_text(yaml.safe_dump(seg, sort_keys=False))


def save_samples(n: int = 12) -> None:
    """Copy a few test frames to dataset/samples/ (small, committed to git for demos)."""
    C.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        src = C.SCENE_DIR / "test" / "images" / f"test_{i:04d}.png"
        if src.exists():
            shutil.copy(src, C.SAMPLE_DIR / f"sample_{i:02d}.png")


def main(argv=None) -> int:
    rules = C.load_rules()
    for d in (C.SCENE_DIR, C.DEFECT_DIR, C.ANNOTATION_DIR):
        if d.exists():
            shutil.rmtree(d)
    summary = {}
    for split, n in C.N_SCENES.items():
        summary[split] = build_split(split, n, rules)
        print(f"{split:5s}: {summary[split]['scenes']} camera frames, {summary[split]['devices']} devices")
    write_yaml_files()
    save_samples()
    print(f"Dataset written to {C.DATASET_DIR.relative_to(C.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
