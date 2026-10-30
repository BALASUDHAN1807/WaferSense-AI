"""Second training stage for the YOLO11n-seg defect segmenter (more effective training on CPU).

    python run_project.py finetune-yoloseg [--epochs 40]

The first stage (python run_project.py train-yoloseg) stopped after 25 epochs while the
validation mask mAP was still rising. Rather than restarting from COCO weights, this stage:

  1. warm-starts from the trained segmenter (models/segmenter_yolo11n_seg.pt);
  2. oversamples crops that contain the weakest classes (scratch x3, missing_pin x2),
     so those thin / small defects are seen more often - a class-balancing step;
  3. uses a lower start learning rate with cosine decay and turns mosaic off for the
     last epochs, the usual recipe for fine-tuning a converged model.

The result is saved as models/segmenter_yolo11n_seg_ft.pt. It replaces the first-stage
model only if the end-to-end decision macro F1 on the VALIDATION split improves
(see compare_and_adopt), so the test split is still used once for the final report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import yaml

from qc import config as C

OVERSAMPLE = {1: 3, 4: 2}            # class id -> copies (scratch, missing_pin)
FT_WEIGHTS = C.MODEL_DIR / "segmenter_yolo11n_seg_ft.pt"
RUN = "yolo_seg_ft"


def balanced_list() -> tuple:
    d = C.DEFECT_DIR
    lines, counts = [], {}
    for img in sorted((d / "train" / "images").glob("*.png")):
        lbl = d / "train" / "labels" / f"{img.stem}.txt"
        classes = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()} if lbl.exists() else set()
        k = max([OVERSAMPLE.get(c, 1) for c in classes] or [1])
        lines += [str(img.resolve())] * k
        for c in classes:
            counts[c] = counts.get(c, 0) + k
    lst = d / "train_balanced.txt"
    lst.write_text("\n".join(lines) + "\n")
    cfg = yaml.safe_load((d / "data.yaml").read_text())
    cfg.update(path=str(d.resolve()), train=str(lst.resolve()))
    yml = d / "data_balanced.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    return yml, len(lines), counts


def finetune(epochs: int = 40) -> dict:
    from ultralytics import YOLO
    if not C.YOLO_SEG_WEIGHTS.exists():
        raise FileNotFoundError("Train the first stage first: python run_project.py train-yoloseg")
    yml, n, counts = balanced_list()
    print(f"Balanced training list: {n} entries, objects per class (after oversampling): {counts}")
    t0 = time.perf_counter()
    YOLO(str(C.YOLO_SEG_WEIGHTS)).train(
        data=str(yml), imgsz=C.YOLO_SEG["imgsz"], epochs=epochs, batch=C.YOLO_SEG["batch"], patience=15,
        lr0=0.002, lrf=0.05, cos_lr=True, warmup_epochs=1, close_mosaic=8, optimizer="AdamW",
        flipud=0.5, fliplr=0.5, degrees=15, mosaic=0.5, device=C.DEVICE, workers=2, seed=C.SEED,
        deterministic=True, project=str(C.RUNS_DIR), name=RUN, exist_ok=True, plots=True, verbose=False)
    shutil.copy(C.RUNS_DIR / RUN / "weights" / "best.pt", FT_WEIGHTS)

    import pandas as pd
    h = pd.read_csv(C.RUNS_DIR / RUN / "results.csv")
    h.columns = [c.strip() for c in h.columns]
    first = pd.read_csv(C.RUNS_DIR / "yolo_seg" / "results.csv")
    first.columns = [c.strip() for c in first.columns]
    k = "metrics/mAP50(M)"
    out = {"stage": "fine-tune", "warm_start": str(C.YOLO_SEG_WEIGHTS.relative_to(C.ROOT)),
           "oversample": {C.DEFECT_CLASSES[c]: v for c, v in OVERSAMPLE.items()},
           "epochs_completed": int(len(h)), "training_time_seconds": round(time.perf_counter() - t0, 1),
           "val_mask_mAP50_stage1_best": float(first[k].max()), "val_mask_mAP50_stage2_best": float(h[k].max()),
           "val_mask_mAP50_95_stage1_best": float(first["metrics/mAP50-95(M)"].max()),
           "val_mask_mAP50_95_stage2_best": float(h["metrics/mAP50-95(M)"].max())}
    (C.REPORT_DIR / "finetune_yoloseg.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    finetune(p.parse_args(argv).epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
