"""Upgrade of the real-image decision rule + uncertainty of every real-image number.

    python -m qc.real.fusion          (needs results of `python run_project.py real-eval`)

1. Score fusion. The original rule lets PatchCore alone decide "defect or not", so a
   defect PatchCore misses is passed even when YOLO11-seg sees it clearly. Fusion:

        defective  =  PatchCore score >= t_pc   OR   best YOLO11-seg confidence >= t_seg

   Both thresholds are chosen jointly on the CALIBRATION split only (best macro F1
   of the PASS / REWORK / SCRAP decision); the held-out split is used once to report.
   YOLO11-seg predictions are the out-of-fold ones from the 5-fold cross-validation,
   so no image is judged by a model that saw it.

2. Bootstrap 95 % confidence intervals (2,000 resamples of the held-out images) for
   accuracy, macro F1, defect recall and false-reject rate of both rules, and for
   PatchCore's AUROC - with only 60 held-out images the intervals are wide and the
   report should say so.
"""
from __future__ import annotations

import json

import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from qc import config as C
from qc.real.evaluate_real import DECISIONS, GT_DECISION, best_threshold, run_patchcore
from qc.real.mvtec import load_samples, split_test

N_BOOT = 2000
SEG_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]      # 1.01 = never (PatchCore only)


def decide(score, dets, t_pc, t_seg, rules):
    top = max(dets, key=lambda d: d["conf"]) if dets else None
    if score >= t_pc or (top is not None and top["conf"] >= t_seg):
        return rules[top["cls"] if top else "unknown"]["verdict"]
    return "PASS"


def metrics(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    dt, dp = yt != "PASS", yp != "PASS"
    return {"accuracy": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, labels=DECISIONS, average="macro", zero_division=0)),
            "defect_recall": float(dp[dt].mean()) if dt.any() else float("nan"),
            "false_reject_rate": float(dp[~dt].mean()) if (~dt).any() else float("nan")}


def bootstrap(yt, yp, seed=C.SEED):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(yt), np.asarray(yp)
    vals = {k: [] for k in metrics(yt, yp)}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt))
        for k, v in metrics(yt[i], yp[i]).items():
            vals[k].append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def auroc_ci(y, s, seed=C.SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].all() or not y[i].any():
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main(argv=None) -> int:
    rules = yaml.safe_load((C.ROOT / "configs" / "transistor_rules.yaml").read_text())
    oof = json.loads((C.REPORT_DIR / "real_supervised_coco.json").read_text())["oof"]
    samples = load_samples()
    calib, evals = split_test(samples)
    test, scores, _ = run_patchcore(samples)
    S = {str(s.path): float(sc) for s, sc in zip(test, scores)}
    D = {k: v["detections"] for k, v in oof.items()}

    def run(items, t_pc, t_seg):
        yt = [GT_DECISION[s.label] for s in items]
        yp = [decide(S[str(s.path)], D[str(s.path)], t_pc, t_seg, rules) for s in items]
        return yt, yp

    # PatchCore threshold candidates from the calibration split (+ the original F1 threshold)
    cal_scores = np.array([S[str(s.path)] for s in calib])
    t_orig = best_threshold(cal_scores, np.array([s.defective for s in calib]))
    pc_grid = sorted(set(np.round(np.quantile(cal_scores, np.linspace(0.3, 0.95, 27)), 4)) | {t_orig})
    best = None
    for t_pc in pc_grid:
        for t_seg in SEG_GRID:
            m = metrics(*run(calib, t_pc, t_seg))
            key = (round(m["macro_f1"], 6), round(m["accuracy"], 6), -t_seg, t_pc)   # ties: prefer simpler/stricter
            if best is None or key > best[0]:
                best = (key, t_pc, t_seg)
    _, t_pc, t_seg = best

    out = {"description": "PatchCore OR YOLO11-seg (out-of-fold, COCO init); thresholds chosen on the "
                          "calibration split; bootstrap 95 % CIs over the held-out split",
           "n_bootstrap": N_BOOT, "heldout_images": len(evals)}
    for name, (a, b) in {"patchcore_only": (t_orig, 1.01), "fusion": (t_pc, t_seg)}.items():
        yt, yp = run(evals, a, b)
        out[name] = {"patchcore_threshold": float(a), "seg_conf_threshold": (None if b > 1 else float(b)),
                     "heldout": metrics(yt, yp), "ci95": bootstrap(yt, yp),
                     "confusion_matrix": {"labels": DECISIONS,
                                          "matrix": confusion_matrix(yt, yp, labels=DECISIONS).tolist()},
                     "per_type_correct": {t: float(np.mean([p == GT_DECISION[t] for s, p in zip(evals, yp)
                                                            if s.label == t]))
                                          for t in GT_DECISION}}
    y = np.array([s.defective for s in evals])
    sc = np.array([S[str(s.path)] for s in evals])
    out["patchcore_auroc_heldout_ci95"] = auroc_ci(y, sc)

    path = C.REPORT_DIR / "real_evaluation.json"
    real = json.loads(path.read_text())
    real["fusion"] = out
    path.write_text(json.dumps(real, indent=2))
    for k in ("patchcore_only", "fusion"):
        r = out[k]
        print(k, "t_pc", round(r["patchcore_threshold"], 3), "t_seg", r["seg_conf_threshold"],
              {m: round(v, 3) for m, v in r["heldout"].items()},
              {m: [round(x, 3) for x in v] for m, v in r["ci95"].items()}, r["confusion_matrix"]["matrix"],
              r["per_type_correct"])
    print("AUROC CI", out["patchcore_auroc_heldout_ci95"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
