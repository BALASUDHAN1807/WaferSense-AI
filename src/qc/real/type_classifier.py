"""Defect-TYPE classifier for real MVTec transistors (improves on YOLO11-seg's type).

    python -m qc.real.type_classifier        (after `python run_project.py real-eval`)

Why: the type only has to be named for images already flagged as defective, and the
labels are per image. Segmenting a thin bent lead from 8 examples is hard; classifying
the whole image (or the region PatchCore points at) is an easier learning problem.

Variants (both fine-tune ImageNet YOLO11s-cls, 224 px, same 5 stratified folds as the
YOLO11-seg cross-validation, so every prediction is out-of-fold):
    full   the whole photo
    crop   a square around PatchCore's hottest point (40 % of the image side), so the
           classifier looks at the defect itself
The variant is chosen on the CALIBRATION split; the held-out split is reported with the
fusion detection rule from qc.real.fusion (score fusion decides defect or not; the
chosen classifier names the type -> REWORK / SCRAP).
"""
from __future__ import annotations

import json
import shutil
import time

import cv2
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

from qc import config as C
from qc.real.evaluate_real import DECISIONS, GT_DECISION, run_patchcore
from qc.real.fusion import bootstrap, metrics
from qc.real.mvtec import DEFECT_TYPES, load_samples, split_test
from qc.real.supervised import folds

WORK = C.DATASET_DIR / "real" / "cls_cv"
EPOCHS = 40
CROP_FRAC = 0.40
IMGSZ = 224


def crop_around(path, amap, frac=CROP_FRAC):
    im = cv2.imread(str(path))
    H, W = im.shape[:2]
    y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
    cy, cx = (y + 0.5) / amap.shape[0] * H, (x + 0.5) / amap.shape[1] * W
    s = int(frac * max(H, W))
    x0 = int(np.clip(cx - s / 2, 0, W - s)); y0 = int(np.clip(cy - s / 2, 0, H - s))
    return im[y0:y0 + s, x0:x0 + s]


def build(variant, defects, maps, fold_of, f):
    d = WORK / variant / f"fold{f}"
    if d.exists():
        shutil.rmtree(d)
    for s in defects:
        split = "val" if fold_of[s.path] == f else "train"
        out = d / split / s.label
        out.mkdir(parents=True, exist_ok=True)
        im = crop_around(s.path, maps[str(s.path)]) if variant == "crop" else cv2.imread(str(s.path))
        cv2.imwrite(str(out / f"{s.path.stem}.png"), cv2.resize(im, (288, 288), interpolation=cv2.INTER_AREA))
    for split in ("train", "val"):                      # every class folder must exist
        for t in DEFECT_TYPES:
            (d / split / t).mkdir(parents=True, exist_ok=True)
    return d


def run_variant(variant, defects, maps, fold_of):
    from ultralytics import YOLO
    pred, t0 = {}, time.perf_counter()
    for f in range(5):
        d = build(variant, defects, maps, fold_of, f)
        model = YOLO(str(C.MODEL_DIR / "pretrained" / "yolo11s-cls.pt"))
        model.train(data=str(d), imgsz=IMGSZ, epochs=EPOCHS, batch=8, patience=0, device=C.DEVICE, workers=2,
                    seed=C.SEED, deterministic=True, project=str(C.RUNS_DIR), name=f"real_cls_{variant}_f{f}",
                    exist_ok=True, verbose=False, plots=False, lr0=0.002, optimizer="AdamW",
                    degrees=10, fliplr=0.5, flipud=0.0, erasing=0.2)
        last = YOLO(str(C.RUNS_DIR / f"real_cls_{variant}_f{f}" / "weights" / "last.pt"))
        for s in [s for s in defects if fold_of[s.path] == f]:
            im = crop_around(s.path, maps[str(s.path)]) if variant == "crop" else cv2.imread(str(s.path))
            r = last.predict(im, imgsz=IMGSZ, device=C.DEVICE, verbose=False)[0]
            names = r.names
            probs = r.probs.data.numpy()
            pred[str(s.path)] = {"cls": names[int(probs.argmax())], "conf": float(probs.max()),
                                 "probs": {names[i]: float(p) for i, p in enumerate(probs)}}
        print(f"[{variant}] fold {f} done", flush=True)
    acc = float(np.mean([pred[str(s.path)]["cls"] == s.label for s in defects]))
    return pred, acc, round(time.perf_counter() - t0, 1)


def main(argv=None) -> int:
    rules = yaml.safe_load((C.ROOT / "configs" / "transistor_rules.yaml").read_text())
    real = json.loads((C.REPORT_DIR / "real_evaluation.json").read_text())
    fz = real["fusion"]["fusion"]
    seg = {k: v["detections"] for k, v in
           json.loads((C.REPORT_DIR / "real_supervised_coco.json").read_text())["oof"].items()}
    samples = load_samples()
    calib, evals = split_test(samples)
    test, scores, maps_arr = run_patchcore(samples)
    S = {str(s.path): float(x) for s, x in zip(test, scores)}
    maps = {str(s.path): m for s, m in zip(test, maps_arr)}
    _, fold_of = folds(samples)
    defects = [s for s in test if s.defective]

    def seg_type(p):
        d = seg[p]
        return max(d, key=lambda x: x["conf"])["cls"] if d else "unknown"

    variants = {"yolo_seg": ({str(s.path): {"cls": seg_type(str(s.path))} for s in defects}, None, None)}
    for v in ("full", "crop"):
        variants[v] = run_variant(v, defects, maps, fold_of)

    def decisions(items, types):
        yt, yp = [], []
        for s in items:
            p = str(s.path)
            top = max(seg[p], key=lambda x: x["conf"]) if seg[p] else None
            flagged = S[p] >= fz["patchcore_threshold"] or (top is not None and top["conf"] >= fz["seg_conf_threshold"])
            yt.append(GT_DECISION[s.label])
            if not flagged:
                yp.append("PASS")
            else:
                t = types[p]["cls"] if p in types else (top["cls"] if top else "unknown")
                yp.append(rules[t]["verdict"])
        return yt, yp

    out = {"description": "Defect type for real transistors: YOLO11-seg (top detection) vs YOLO11s-cls on the "
                          "full photo or on a PatchCore-guided crop; out-of-fold, 5 stratified folds; detection "
                          "= score fusion (report 12.4); variant chosen on the calibration split",
           "epochs": EPOCHS, "imgsz": IMGSZ, "crop_fraction": CROP_FRAC, "variants": {}}
    for name, (types, _, secs) in variants.items():
        acc = float(np.mean([types[str(s.path)]["cls"] == s.label for s in defects]))
        per = {t: float(np.mean([types[str(s.path)]["cls"] == t for s in defects if s.label == t])) for t in DEFECT_TYPES}
        cal = metrics(*decisions(calib, types))
        yt, yp = decisions(evals, types)
        out["variants"][name] = {"type_accuracy_oof": acc, "per_type_accuracy_oof": per,
                                 "calibration_macro_f1": cal["macro_f1"], "heldout": metrics(yt, yp),
                                 "ci95": bootstrap(yt, yp), "training_time_seconds": secs,
                                 "confusion_matrix": {"labels": DECISIONS,
                                                      "matrix": confusion_matrix(yt, yp, labels=DECISIONS).tolist()}}
        print(name, round(acc, 3), {k: round(v, 2) for k, v in per.items()}, "calib F1", round(cal["macro_f1"], 3),
              "heldout", {k: round(v, 3) for k, v in metrics(yt, yp).items()})
    cal_def = [s for s in calib if s.defective]
    cal_acc = {k: float(np.mean([variants[k][0][str(s.path)]["cls"] == s.label for s in cal_def])) for k in variants}
    chosen = max(out["variants"], key=lambda k: (out["variants"][k]["calibration_macro_f1"], cal_acc[k]))
    out["chosen"] = chosen
    real["type_classifier"] = out
    (C.REPORT_DIR / "real_evaluation.json").write_text(json.dumps(real, indent=2))
    print("chosen on calibration split:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
