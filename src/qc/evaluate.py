"""Evaluation on the held-out TEST split (never used for training or model selection).

1. Detector         box precision / recall / mAP50 / mAP50-95             (Ultralytics val)
2. YOLO11-seg       mask mAP50 / mAP50-95, overall and per defect class    (Ultralytics val)
3. Both segmenters  pixel IoU and Dice per defect class on the test crops
4. End-to-end       full pipeline on every test camera frame for each segmenter:
                    device matching (IoU >= 0.5), Pass/Rework/Scrap accuracy, macro F1,
                    confusion matrix, defect-type recognition per device, latency
5. Comparison       best segmenter = highest end-to-end decision macro F1
                    (ties: accuracy, then speed) -> results/reports/best_pipeline.json

Usage:
    python run_project.py evaluate
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from qc import config as C

SEGMENTERS = {"yolo": "YOLO11n-seg", "unet": "U-Net"}


def _save(obj, name):
    C.ensure_dirs()
    (C.REPORT_DIR / name).write_text(json.dumps(obj, indent=2, default=float))


# ------------------------------------------------------------ 1 + 2: YOLO val

def eval_detector() -> dict:
    from ultralytics import YOLO
    m = YOLO(str(C.DETECTOR_WEIGHTS)).val(data=str(C.SCENE_DIR / "data.yaml"), split="test",
                                          imgsz=C.DETECTOR["imgsz"], device="cpu", verbose=False, plots=False,
                                          project=str(C.RUNS_DIR), name="eval_detector", exist_ok=True)
    out = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
           "mAP50_95": float(m.box.map), "test_images": C.N_SCENES["test"]}
    _save(out, "detector_test_metrics.json")
    return out


def eval_yolo_seg() -> dict:
    from ultralytics import YOLO
    m = YOLO(str(C.YOLO_SEG_WEIGHTS)).val(data=str(C.DEFECT_DIR / "data.yaml"), split="test",
                                          imgsz=C.YOLO_SEG["imgsz"], device="cpu", verbose=False, plots=False,
                                          project=str(C.RUNS_DIR), name="eval_yolo_seg", exist_ok=True)
    per_class = {}
    for i, c in enumerate(m.seg.ap_class_index):
        per_class[C.DEFECT_CLASSES[int(c)]] = {"mask_mAP50": float(m.seg.ap50[i]), "mask_mAP50_95": float(m.seg.ap[i])}
    out = {"mask_precision": float(m.seg.mp), "mask_recall": float(m.seg.mr),
           "mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map),
           "box_mAP50": float(m.box.map50), "per_class": per_class}
    _save(out, "yolo_seg_test_metrics.json")
    return out


# --------------------------------------------------- 3: pixel IoU / Dice

def semantic_prediction(engine, crop) -> np.ndarray:
    """Segmenter output as a class-index map (0 = background); higher confidence paints last."""
    pred = np.zeros(crop.shape[:2], np.uint8)
    for cls, p, mask in sorted(engine.segment(crop), key=lambda t: t[1]):
        pred[mask] = C.DEFECT_TO_ID[cls] + 1
    return pred


def eval_pixels(engine) -> dict:
    from qc.train import iou_dice_from_confusion
    n = len(C.DEFECT_CLASSES) + 1
    cm = np.zeros((n, n), np.int64)
    d = C.DEFECT_DIR / "test"
    t0 = time.perf_counter()
    files = sorted((d / "images").glob("*.png"))
    for f in files:
        crop = cv2.imread(str(f))
        gt = cv2.imread(str(d / "masks" / f.name), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        pred = semantic_prediction(engine, crop).astype(np.int64)
        cm += np.bincount(gt.ravel() * n + pred.ravel(), minlength=n * n).reshape(n, n)
    ms = (time.perf_counter() - t0) / len(files) * 1000
    iou, dice = iou_dice_from_confusion(cm)
    return {"per_class": {c: {"IoU": float(iou[i + 1]), "Dice": float(dice[i + 1])}
                          for i, c in enumerate(C.DEFECT_CLASSES)},
            "mean_IoU": float(np.nanmean(iou[1:])), "mean_Dice": float(np.nanmean(dice[1:])),
            "background_IoU": float(iou[0]), "ms_per_crop": round(ms, 2), "test_crops": len(files)}


# ------------------------------------------------------- 4: end to end

def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, pred_boxes, thr=0.5):
    """Greedy one-to-one matching by IoU -> list of (gt_idx, pred_idx)."""
    pairs = sorted(((box_iou(g, p), i, j) for i, g in enumerate(gt_boxes) for j, p in enumerate(pred_boxes)),
                   reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def eval_end_to_end(engine, save_examples: int = 6) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from qc.engine import annotate

    ann = json.loads((C.ANNOTATION_DIR / "test.json").read_text())
    y_true, y_pred, latencies = [], [], []
    missed = false_dev = 0
    type_tp = {c: 0 for c in C.DEFECT_CLASSES}
    type_fn = dict(type_tp)
    type_fp = dict(type_tp)
    for k, scene in enumerate(ann["scenes"]):
        img = cv2.imread(str(C.DATASET_DIR / scene["image"]))
        res = engine.inspect(img)
        latencies.append(res.timings_ms["total"])
        gts = scene["devices"]
        pairs = match([g["box"] for g in gts], [d.box for d in res.devices])
        missed += len(gts) - len(pairs)
        false_dev += len(res.devices) - len(pairs)
        matched_g = {i for i, _ in pairs}
        for i, g in enumerate(gts):            # a missed device counts as a wrong decision
            if i not in matched_g:
                y_true.append(g["decision"])
                y_pred.append("MISSED")
        for i, j in pairs:
            g, p = gts[i], res.devices[j]
            y_true.append(g["decision"])
            y_pred.append(p.decision)
            # defect-type recognition: which significant (non-PASS) defect types are present
            gt_types = {d["class"] for d in g["defects"] if d["verdict"] != "PASS"}
            pr_types = {d.cls for d in p.defects if d.verdict != "PASS"}
            for c in C.DEFECT_CLASSES:
                type_tp[c] += c in gt_types and c in pr_types
                type_fn[c] += c in gt_types and c not in pr_types
                type_fp[c] += c not in gt_types and c in pr_types
        if k < save_examples:
            C.PRED_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(C.PRED_DIR / f"{engine.segmenter}_test_{k:02d}.png"), annotate(img, res))

    labels = C.DECISIONS
    cm = confusion_matrix(y_true, y_pred, labels=labels + ["MISSED"])[:3]
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    types = {}
    for c in C.DEFECT_CLASSES:
        tp, fn, fp = type_tp[c], type_fn[c], type_fp[c]
        types[c] = {"recall": tp / (tp + fn) if tp + fn else None,
                    "precision": tp / (tp + fp) if tp + fp else None, "support": tp + fn}
    return {
        "segmenter": engine.segmenter, "test_frames": len(ann["scenes"]), "gt_devices": len(y_true),
        "missed_devices": missed, "false_devices": false_dev,
        "decision_accuracy": float(accuracy_score(y_true, y_pred)),
        "decision_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_decision": {l: {k: float(rep[l][k]) for k in ("precision", "recall", "f1-score", "support")}
                         for l in labels},
        "confusion_matrix": {"labels_true": labels, "labels_pred": labels + ["MISSED"], "matrix": cm.tolist()},
        "scrap_escape_rate": float(sum(1 for t, p in zip(y_true, y_pred) if t == "SCRAP" and p == "PASS")
                                   / max(1, y_true.count("SCRAP"))),
        "false_scrap_rate": float(sum(1 for t, p in zip(y_true, y_pred) if t == "PASS" and p == "SCRAP")
                                  / max(1, y_true.count("PASS"))),
        "defect_type_recognition": types,
        "latency_ms_per_frame": {"mean": float(np.mean(latencies)), "median": float(np.median(latencies)),
                                 "p95": float(np.percentile(latencies, 95))},
    }


# ------------------------------------------- threshold tuning (VALIDATION)

def tune_thresholds(segmenter: str) -> dict:
    """Pick the defect confidence threshold and minimum defect size (crop pixels) that maximise
    decision macro F1 on the VALIDATION split.  The same grid and procedure are used for both
    segmenters; the test split is not touched."""
    from sklearn.metrics import f1_score

    from qc.engine import InspectionEngine
    from qc.quality import combine

    engine = InspectionEngine(segmenter, seg_conf=min(C.TUNE_SEG_CONF), min_pixels=1)
    ann = json.loads((C.ANNOTATION_DIR / "val.json").read_text())
    records = []                     # (gt decision, [(conf, pixels, verdict), ...] or None if missed)
    for scene in ann["scenes"]:
        res = engine.inspect(cv2.imread(str(C.DATASET_DIR / scene["image"])))
        gts = scene["devices"]
        pairs = dict(match([g["box"] for g in gts], [d.box for d in res.devices]))
        for i, g in enumerate(gts):
            dev = res.devices[pairs[i]] if i in pairs else None
            records.append((g["decision"], None if dev is None else
                            [(d.confidence, d.pixels, d.verdict) for d in dev.defects]))
    grid = []
    y_true = [r[0] for r in records]
    for conf in C.TUNE_SEG_CONF:
        for mp in C.TUNE_MIN_PIXELS:
            y_pred = ["MISSED" if d is None else combine(v for p, n, v in d if p >= conf and n >= mp)
                      for _, d in records]
            f1 = f1_score(y_true, y_pred, labels=C.DECISIONS, average="macro", zero_division=0)
            acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
            grid.append({"seg_conf": conf, "min_pixels": mp, "val_macro_f1": float(f1), "val_accuracy": acc})
    best = max(grid, key=lambda g: (round(g["val_macro_f1"], 4), g["val_accuracy"], -g["seg_conf"]))
    default = next((g for g in grid if g["seg_conf"] == 0.2 and g["min_pixels"] == 6), None)
    return {**best, "val_devices": len(records), "default_0.2_6": default, "grid": grid}


# -------------------------------------------------------------- plots

def _plots(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [SEGMENTERS[s] for s in results["segmenters"]]
    segs = results["segmenters"]

    # confusion matrices
    fig, axes = plt.subplots(1, len(segs), figsize=(5.6 * len(segs), 4.6))
    for ax, s in zip(np.atleast_1d(axes), segs):
        e = results["end_to_end"][s]
        cm = np.array(e["confusion_matrix"]["matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(cm.shape[1]), e["confusion_matrix"]["labels_pred"])
        ax.set_yticks(range(cm.shape[0]), e["confusion_matrix"]["labels_true"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted decision")
        ax.set_ylabel("True decision")
        ax.set_title(f"{SEGMENTERS[s]} pipeline - accuracy {e['decision_accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(C.GRAPH_DIR / "decision_confusion_matrices.png", dpi=120)
    plt.close(fig)

    # per-class IoU
    x = np.arange(len(C.DEFECT_CLASSES))
    w = 0.8 / len(segs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, s in enumerate(segs):
        vals = [results["pixel"][s]["per_class"][c]["IoU"] for c in C.DEFECT_CLASSES]
        bars = ax.bar(x + (k - (len(segs) - 1) / 2) * w, vals, w, label=SEGMENTERS[s])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, C.DEFECT_CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU (test crops)")
    ax.set_title("Defect segmentation IoU per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.GRAPH_DIR / "segmentation_iou_per_class.png", dpi=120)
    plt.close(fig)

    # headline comparison
    metrics = [("decision_accuracy", "Decision accuracy"), ("decision_macro_f1", "Decision macro F1")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for k, s in enumerate(segs):
        e = results["end_to_end"][s]
        axes[0].bar(np.arange(2) + (k - (len(segs) - 1) / 2) * w, [e[m] for m, _ in metrics], w, label=names[k])
    axes[0].set_xticks(range(2), [l for _, l in metrics])
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].set_title("End-to-end quality decision")
    axes[1].bar(names, [results["pixel"][s]["mean_IoU"] for s in segs], color=["#3b6ea5", "#e07b39"][:len(segs)])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Mean defect IoU")
    axes[2].bar(names, [results["end_to_end"][s]["latency_ms_per_frame"]["mean"] for s in segs],
                color=["#3b6ea5", "#e07b39"][:len(segs)])
    axes[2].set_title("Latency per camera frame (ms, CPU)")
    for ax in axes[1:]:
        for b in ax.patches:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.GRAPH_DIR / "segmenter_comparison.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------- main

def select_best(e2e: dict) -> dict:
    ranked = sorted(e2e.values(), key=lambda r: (-round(r["decision_macro_f1"], 3), -r["decision_accuracy"],
                                                 r["latency_ms_per_frame"]["mean"]))
    best = ranked[0]
    reason = (f"{SEGMENTERS[best['segmenter']]} gives the highest end-to-end decision macro F1 "
              f"({best['decision_macro_f1']:.4f}, accuracy {best['decision_accuracy']:.4f})")
    if len(ranked) > 1:
        reason += f" vs {SEGMENTERS[ranked[1]['segmenter']]} ({ranked[1]['decision_macro_f1']:.4f})."
    return {"segmenter": best["segmenter"], "segmenter_name": SEGMENTERS[best["segmenter"]],
            "decision_macro_f1": best["decision_macro_f1"], "decision_accuracy": best["decision_accuracy"],
            "latency_ms_per_frame": best["latency_ms_per_frame"]["mean"], "reason": reason}


def main(argv=None) -> int:
    from qc.engine import InspectionEngine
    C.ensure_dirs()
    results = {"detector": eval_detector()}
    print("Detector:", json.dumps(results["detector"], indent=1))
    segs = [s for s, w in (("yolo", C.YOLO_SEG_WEIGHTS), ("unet", C.UNET_WEIGHTS)) if w.exists()]
    if not segs:
        print("No segmentation model trained yet.")
        return 1
    results["segmenters"] = segs
    if "yolo" in segs:
        results["yolo_seg_val"] = eval_yolo_seg()
        print("YOLO-seg:", json.dumps({k: v for k, v in results["yolo_seg_val"].items() if k != "per_class"}))
    # post-processing thresholds for each segmenter, tuned on the validation split only
    tuned = {}
    for s in segs:
        tuned[s] = tune_thresholds(s)
        print(f"{SEGMENTERS[s]:12s} tuned on validation: conf >= {tuned[s]['seg_conf']}, "
              f"min {tuned[s]['min_pixels']} px  (val macro-F1 {tuned[s]['val_macro_f1']:.4f})")
    _save(tuned, C.THRESHOLDS_FILE.name)
    results["tuned_thresholds"] = {s: {k: v for k, v in t.items() if k != "grid"} for s, t in tuned.items()}

    results["pixel"], results["end_to_end"] = {}, {}
    for s in segs:
        engine = InspectionEngine(s)
        results["pixel"][s] = eval_pixels(engine)
        results["end_to_end"][s] = eval_end_to_end(engine)
        e = results["end_to_end"][s]
        print(f"{SEGMENTERS[s]:12s} mIoU {results['pixel'][s]['mean_IoU']:.4f} | decision acc "
              f"{e['decision_accuracy']:.4f} macro-F1 {e['decision_macro_f1']:.4f} | missed {e['missed_devices']} "
              f"| {e['latency_ms_per_frame']['mean']:.0f} ms/frame")
    results["best"] = select_best(results["end_to_end"])
    _save(results, "evaluation.json")
    _save(results["best"], "best_pipeline.json")
    _plots(results)
    print("\nBest pipeline:", results["best"]["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
