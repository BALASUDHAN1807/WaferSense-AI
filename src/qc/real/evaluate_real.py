"""Evaluation on REAL images (MVTec AD transistor).

1. PatchCore (anomaly detection, trained on defect-free images only)
   - image AUROC and pixel AUROC on all 100 MVTec test images
   - decision threshold chosen on a calibration split (40 % of the test images),
     PASS / DEFECT accuracy reported on the remaining 60 % (held out)
2. Supervised YOLO11n-seg (5-fold cross-validation, out-of-fold predictions)
   - mask mAP50 per fold, defect-type accuracy
3. Combined system on the held-out split:
   PatchCore says PASS -> PASS; otherwise the defect type from YOLO11-seg picks
   REWORK / SCRAP via configs/transistor_rules.yaml.

Usage:
    python run_project.py real-eval [--skip-supervised]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from qc import config as C
from qc.real.mvtec import DEFECT_TYPES, load_samples, split_test
from qc.real.patchcore import IMG_SIZE, PatchCore, load_mask

GT_DECISION = {"good": "PASS", "bent_lead": "REWORK", "misplaced": "REWORK",
               "cut_lead": "SCRAP", "damaged_case": "SCRAP"}
DECISIONS = ["PASS", "REWORK", "SCRAP"]
PATCHCORE_FILE = C.MODEL_DIR / "patchcore_transistor.pt"


def best_threshold(scores, labels):
    """Threshold maximising F1 for 'defective' on the calibration split."""
    cands = np.unique(scores)
    f1s = [f1_score(labels, scores >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def run_patchcore(samples):
    good = [s.path for s in samples if s.origin == "train"]
    test = [s for s in samples if s.origin == "test"]
    pc = PatchCore()
    if PATCHCORE_FILE.exists():
        pc.load(PATCHCORE_FILE)
    else:
        print(f"Fitting PatchCore on {len(good)} defect-free images ...", flush=True)
        pc.fit(good)
        C.ensure_dirs()
        pc.save(PATCHCORE_FILE)
    scores, maps = pc.score([s.path for s in test])
    return test, scores, maps


def pixel_auroc(test, maps):
    gts, preds = [], []
    for s, m in zip(test, maps):
        gt = load_mask(s.mask) if s.mask else np.zeros((IMG_SIZE, IMG_SIZE), bool)
        gts.append(gt.ravel())
        preds.append(m.ravel())
    return float(roc_auc_score(np.concatenate(gts), np.concatenate(preds)))


def save_heatmaps(test, scores, maps, thr, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    picks = []
    for label in ["good"] + DEFECT_TYPES:
        idx = [i for i, s in enumerate(test) if s.label == label]
        picks += idx[:2 if label == "good" else 1][: n]
    picks = picks[:6]
    fig, axes = plt.subplots(3, len(picks), figsize=(2.6 * len(picks), 8))
    vmax = np.percentile(maps, 99.5)
    for j, i in enumerate(picks):
        s = test[i]
        im = Image.open(s.path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        axes[0, j].imshow(im)
        axes[0, j].set_title(f"{s.label}\nscore {scores[i]:.2f} ({'DEFECT' if scores[i] >= thr else 'PASS'})",
                             fontsize=8)
        gt = load_mask(s.mask) if s.mask else np.zeros((IMG_SIZE, IMG_SIZE), bool)
        axes[1, j].imshow(gt, cmap="gray")
        axes[2, j].imshow(im)
        axes[2, j].imshow(maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        for r in range(3):
            axes[r, j].axis("off")
    axes[0, 0].set_ylabel("image")
    fig.suptitle("Real MVTec AD transistors: image / ground-truth defect mask / PatchCore anomaly heat-map")
    fig.tight_layout()
    out = C.GRAPH_DIR / "real_patchcore_heatmaps.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-supervised", action="store_true")
    p.add_argument("--init", nargs="+", default=["synthetic", "coco"])
    a = p.parse_args(argv)
    C.ensure_dirs()
    samples = load_samples()
    calib, evals = split_test(samples)
    test, scores, maps = run_patchcore(samples)
    idx = {str(s.path): i for i, s in enumerate(test)}
    y_def = np.array([s.defective for s in test])

    c_idx = [idx[str(s.path)] for s in calib]
    e_idx = [idx[str(s.path)] for s in evals]
    thr = best_threshold(scores[c_idx], y_def[c_idx])
    pred_def = scores >= thr

    def det_metrics(ix):
        ix = np.array(ix)
        yt, yp = y_def[ix], pred_def[ix]
        return {"n": int(len(ix)), "accuracy": float((yt == yp).mean()),
                "defect_recall": float(yp[yt].mean()), "false_reject_rate": float(yp[~yt].mean()),
                "escapes": int((yt & ~yp).sum())}

    pc = {
        "train_good_images": sum(s.origin == "train" for s in samples),
        "test_images": len(test), "calibration_images": len(calib), "heldout_images": len(evals),
        "image_AUROC_all_test": float(roc_auc_score(y_def, scores)),
        "image_AUROC_heldout": float(roc_auc_score(y_def[e_idx], scores[e_idx])),
        "pixel_AUROC_all_test": pixel_auroc(test, maps),
        "threshold": thr, "heldout": det_metrics(e_idx),
        "per_type_recall_heldout": {t: float(pred_def[[i for i in e_idx if test[i].label == t]].mean())
                                    for t in DEFECT_TYPES},
        "heatmaps": str(save_heatmaps(test, scores, maps, thr).relative_to(C.ROOT)),
    }
    print(json.dumps({k: v for k, v in pc.items() if k != "heatmaps"}, indent=1))
    results = {"dataset": "MVTec AD - transistor (real photographs)", "patchcore": pc}

    if not a.skip_supervised:
        from qc.real.supervised import run_cv
        rules = yaml.safe_load((C.ROOT / "configs" / "transistor_rules.yaml").read_text())
        results["supervised"] = {}
        for init in a.init:
            cv = run_cv(init)
            oof = cv["oof"]
            # defect type = highest-confidence detection
            type_true, type_pred = [], []
            for s in test:
                if not s.defective:
                    continue
                dets = oof[str(s.path)]["detections"]
                type_true.append(s.label)
                type_pred.append(max(dets, key=lambda d: d["conf"])["cls"] if dets else "none")
            # combined decision on the held-out split
            yt, yp = [], []
            for i in e_idx:
                s = test[i]
                yt.append(GT_DECISION[s.label])
                if not pred_def[i]:
                    yp.append("PASS")
                else:
                    dets = oof[str(s.path)]["detections"]
                    t = max(dets, key=lambda d: d["conf"])["cls"] if dets else "unknown"
                    yp.append(rules[t]["verdict"])
            cm = confusion_matrix(yt, yp, labels=DECISIONS)
            results["supervised"][init] = {
                "mean_mask_mAP50": cv["mean_mask_mAP50"], "std_mask_mAP50": cv["std_mask_mAP50"],
                "folds": cv["folds"], "training_time_seconds": cv["training_time_seconds"],
                "defect_type_accuracy_oof": float(np.mean([t == p for t, p in zip(type_true, type_pred)])),
                "combined_decision_accuracy_heldout": float(np.mean([t == p for t, p in zip(yt, yp)])),
                "combined_decision_macro_f1_heldout": float(f1_score(yt, yp, labels=DECISIONS, average="macro",
                                                                     zero_division=0)),
                "combined_confusion_matrix": {"labels": DECISIONS, "matrix": cm.tolist()},
            }
            print(init, json.dumps({k: v for k, v in results["supervised"][init].items() if k != "folds"}))
    (C.REPORT_DIR / "real_evaluation.json").write_text(json.dumps(results, indent=2))
    print("Saved results/reports/real_evaluation.json")
    if "coco" in results.get("supervised", {}):
        from qc.real.fusion import main as fusion_main      # score fusion + bootstrap CIs (report 12.4-12.5)
        fusion_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
