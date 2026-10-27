"""Build the real-image (MVTec transistor) models for the web demo in deploy/vercel/.

    python -m qc.real.export_demo

1. Picks demo images from MVTec's test set (2 good + 1 per defect type) that are
   kept OUT of every model trained here, so the demo shows unseen images.
2. Trains the final defect-type model: YOLO11n-seg (COCO init, 384 px) on all other
   MVTec test images + 60 defect-free images (same recipe as the 5-fold CV).
3. Exports to deploy/vercel/models/:
     patchcore_backbone.onnx     PatchCore feature extractor (YOLO11s-cls stages 0-6)
     patchcore_bank.npy          memory bank of defect-free patch features (float16)
     transistor_seg.onnx         defect-type model
   and writes the demo images + settings into deploy/vercel/.
"""
from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import torch
import yaml

from qc import config as C
from qc.real.mvtec import DEFECT_TYPES, load_samples, split_test
from qc.real.patchcore import IMG_SIZE, PatchCore
from qc.real.supervised import IMGSZ, N_GOOD_TRAIN, _write, init_weights

DEPLOY = C.ROOT / "deploy" / "vercel"
WORK = C.DATASET_DIR / "real" / "yolo_final"
PC_MODEL = C.MODEL_DIR / "patchcore_transistor.pt"
FINAL_WEIGHTS = C.MODEL_DIR / "transistor_seg_yolo11n.pt"


def pick_demo(samples):
    """Deterministic demo set from the held-out split: 2 good + 1 per defect type."""
    _, heldout = split_test(samples)
    demo = [s for s in heldout if s.label == "good"][:2]
    for t in DEFECT_TYPES:
        demo.append(next(s for s in heldout if s.label == t))
    return demo


def train_final(samples, demo, epochs=40):
    from ultralytics import YOLO
    if FINAL_WEIGHTS.exists():
        return FINAL_WEIGHTS
    if WORK.exists():
        shutil.rmtree(WORK)
    rng = np.random.default_rng(C.SEED)
    goods = [s for s in samples if s.origin == "train"]
    neg = [goods[i] for i in rng.choice(len(goods), N_GOOD_TRAIN, replace=False)]
    demo_paths = {s.path for s in demo}
    train = [s for s in samples if s.origin == "test" and s.path not in demo_paths] + neg
    # Ultralytics needs a val split; a few training images are reused (last.pt is kept, not "best")
    for split, items in (("train", train), ("val", train[::10])):
        (WORK / split / "images").mkdir(parents=True)
        (WORK / split / "labels").mkdir(parents=True)
        for s in items:
            name = f"{s.label}_{s.origin}_{s.path.stem}"
            _write(s, WORK / split / "images" / f"{name}.png", WORK / split / "labels" / f"{name}.txt")
    (WORK / "data.yaml").write_text(yaml.safe_dump({"path": str(WORK.resolve()), "train": "train/images",
                                                    "val": "val/images", "names": dict(enumerate(DEFECT_TYPES))}))
    model = YOLO(init_weights("coco"))
    model.train(data=str(WORK / "data.yaml"), imgsz=IMGSZ, epochs=epochs, batch=8, patience=0,
                device=C.DEVICE, workers=2, seed=C.SEED, deterministic=True, project=str(C.RUNS_DIR),
                name="real_final", exist_ok=True, verbose=False, plots=False,
                degrees=10, fliplr=0.5, mosaic=0.5)
    shutil.copy(C.RUNS_DIR / "real_final" / "weights" / "last.pt", FINAL_WEIGHTS)
    return FINAL_WEIGHTS


def export_patchcore(threshold: float):
    pc = PatchCore().load(PC_MODEL)
    out = DEPLOY / "models"
    torch.onnx.export(pc.backbone, torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), str(out / "patchcore_backbone.onnx"),
                      input_names=["images"], output_names=["features"], opset_version=17, dynamo=False)
    np.save(out / "patchcore_bank.npy", pc.bank.numpy().astype(np.float16))
    return {"img_size": IMG_SIZE, "threshold": threshold, "blur_sigma": 4}


def main(argv=None) -> int:
    samples = load_samples()
    demo = pick_demo(samples)
    real = json.loads((C.REPORT_DIR / "real_evaluation.json").read_text())
    fus = real.get("fusion", {}).get("fusion")          # python -m qc.real.fusion
    cfg = {"patchcore": export_patchcore(fus["patchcore_threshold"] if fus else real["patchcore"]["threshold"])}

    from ultralytics import YOLO
    w = train_final(samples, demo)
    onnx = YOLO(str(w)).export(format="onnx", imgsz=IMGSZ, opset=17, simplify=False)
    shutil.copy(onnx, DEPLOY / "models" / "transistor_seg.onnx")
    rules = yaml.safe_load((C.ROOT / "configs" / "transistor_rules.yaml").read_text())
    cfg["seg"] = {"img_size": IMGSZ, "classes": DEFECT_TYPES, "conf": 0.10,
                 "gate_conf": fus["seg_conf_threshold"] if fus else None}
    cfg["rules"] = rules

    pub = DEPLOY / "public" / "real"
    if pub.exists():
        shutil.rmtree(pub)
    pub.mkdir(parents=True)
    cfg["samples"] = []
    for s in demo:
        name = f"{s.label}_{s.path.stem}.jpg"
        im = cv2.resize(cv2.imread(str(s.path)), (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(pub / name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cfg["samples"].append({"file": f"real/{name}", "label": s.label})
    (DEPLOY / "real_config.json").write_text(json.dumps(cfg, indent=2))
    print("Exported real-image demo models to", DEPLOY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
