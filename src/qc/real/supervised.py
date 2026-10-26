"""Supervised defect segmentation on REAL images (MVTec AD transistor) with YOLO11-seg.

MVTec provides only 40 defective transistor images (10 per type), so the model is
evaluated with stratified K-fold cross-validation over all 100 MVTec test images:
every image is predicted by a model that never saw it during training
("out-of-fold" predictions). Defect-free MVTec training images are added to every
fold as negative (background) examples.

Two initialisations are compared:
    coco       YOLO11n-seg pretrained on COCO (generic)
    synthetic  our YOLO11n-seg trained on the synthetic IC dataset (transfer sim -> real)
"""
from __future__ import annotations

import json
import shutil
import time

import cv2
import numpy as np
import yaml

from qc import config as C
from qc.real.mvtec import DEFECT_TYPES, load_samples

IMGSZ = 384
EPOCHS = 40
N_FOLDS = 5
N_GOOD_TRAIN = 60            # defect-free MVTec train images added per fold as negatives
WORK = C.DATASET_DIR / "real" / "yolo_cv"


def _write(sample, dst_img, dst_lbl):
    im = cv2.imread(str(sample.path))
    im = cv2.resize(im, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_img), im)
    lines = []
    if sample.mask is not None:
        m = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST) > 127
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 4 or len(c) < 3:
                continue
            pts = (c.reshape(-1, 2) / IMGSZ).clip(0, 1)
            lines.append(f"{DEFECT_TYPES.index(sample.label)} " + " ".join(f"{v:.5f}" for v in pts.ravel()))
    dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))


def folds(samples, k=N_FOLDS, seed=C.SEED):
    """Stratified folds over MVTec's test images (good + 4 defect types)."""
    rng = np.random.default_rng(seed)
    test = [s for s in samples if s.origin == "test"]
    fold_of = {}
    for label in sorted({s.label for s in test}):
        group = [s for s in test if s.label == label]
        for j, i in enumerate(rng.permutation(len(group))):
            fold_of[group[i].path] = j % k
    return test, fold_of


def build_fold(samples, test, fold_of, f, seed=C.SEED):
    d = WORK / f"fold{f}"
    if d.exists():
        shutil.rmtree(d)
    rng = np.random.default_rng(seed + f)
    goods = [s for s in samples if s.origin == "train"]
    neg = [goods[i] for i in rng.choice(len(goods), N_GOOD_TRAIN, replace=False)]
    train = [s for s in test if fold_of[s.path] != f] + neg
    val = [s for s in test if fold_of[s.path] == f]
    for split, items in (("train", train), ("val", val)):
        (d / split / "images").mkdir(parents=True)
        (d / split / "labels").mkdir(parents=True)
        for k, s in enumerate(items):
            name = f"{s.label}_{s.origin}_{s.path.stem}"
            _write(s, d / split / "images" / f"{name}.png", d / split / "labels" / f"{name}.txt")
    (d / "data.yaml").write_text(yaml.safe_dump({"path": str(d.resolve()), "train": "train/images",
                                                 "val": "val/images", "names": dict(enumerate(DEFECT_TYPES))}))
    return d, val


def init_weights(init: str):
    if init == "coco":
        from qc.train import pretrained
        return str(pretrained("yolo11n-seg.pt"))
    if init == "synthetic":
        return str(C.YOLO_SEG_WEIGHTS)
    raise ValueError(init)


def run_cv(init: str = "synthetic", epochs: int = EPOCHS) -> dict:
    """Train K fold models; return out-of-fold predictions for every MVTec test image."""
    from ultralytics import YOLO

    samples = load_samples()
    test, fold_of = folds(samples)
    oof, fold_metrics, t0 = {}, [], time.perf_counter()
    for f in range(N_FOLDS):
        d, val = build_fold(samples, test, fold_of, f)
        model = YOLO(init_weights(init))
        model.train(data=str(d / "data.yaml"), imgsz=IMGSZ, epochs=epochs, batch=8, patience=15,
                    device=C.DEVICE, workers=2, seed=C.SEED, deterministic=True, project=str(C.RUNS_DIR),
                    name=f"real_{init}_fold{f}", exist_ok=True, verbose=False, plots=False,
                    degrees=10, fliplr=0.5, mosaic=0.5)
        best = YOLO(str(C.RUNS_DIR / f"real_{init}_fold{f}" / "weights" / "best.pt"))
        m = best.val(data=str(d / "data.yaml"), imgsz=IMGSZ, device=C.DEVICE, verbose=False, plots=False)
        fold_metrics.append({"fold": f, "mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map),
                             "box_mAP50": float(m.box.map50)})
        for s in val:
            r = best.predict(str(s.path), imgsz=IMGSZ, conf=0.1, device=C.DEVICE, verbose=False, retina_masks=True)[0]
            dets = []
            if r.masks is not None:
                for c, p, mk in zip(r.boxes.cls.numpy(), r.boxes.conf.numpy(), r.masks.data.numpy()):
                    dets.append({"cls": DEFECT_TYPES[int(c)], "conf": float(p), "pixels": int((mk > 0.5).sum())})
            oof[str(s.path)] = {"label": s.label, "fold": f, "detections": dets}
        print(f"[{init}] fold {f}: mask mAP50 {fold_metrics[-1]['mask_mAP50']:.3f}", flush=True)
    result = {"init": init, "folds": fold_metrics, "epochs": epochs, "imgsz": IMGSZ,
              "mean_mask_mAP50": float(np.mean([x["mask_mAP50"] for x in fold_metrics])),
              "std_mask_mAP50": float(np.std([x["mask_mAP50"] for x in fold_metrics])),
              "training_time_seconds": round(time.perf_counter() - t0, 1), "oof": oof}
    C.ensure_dirs()
    (C.REPORT_DIR / f"real_supervised_{init}.json").write_text(json.dumps(result, indent=1))
    return result
