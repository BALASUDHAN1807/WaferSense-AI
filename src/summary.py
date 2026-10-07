"""Collect the ACTUAL generated results into results/reports/final_project_summary.json.

Usage:
    python src/summary.py
"""
from __future__ import annotations

import platform
import sys

import pandas as pd

import config as C
from utils import load_json, save_json


def _maybe(path):
    return load_json(path) if path.exists() else None


def build_summary() -> dict:
    eda = _maybe(C.REPORT_DIR / "eda_summary.json")
    results = _maybe(C.REPORT_DIR / "all_model_results.json")
    best = _maybe(C.BEST_MODEL_FILE)
    split = pd.read_csv(C.SPLIT_FILE, keep_default_na=False) if C.SPLIT_FILE.exists() else None

    models = {}
    for r in (results or {}).get("results", []):
        models[r["model"]] = {
            "accuracy": round(r["accuracy"], 4),
            "macro_precision": round(r["macro_precision"], 4),
            "macro_recall": round(r["macro_recall"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "training_time_seconds": r.get("training_time_seconds"),
            "inference_time_ms_per_image": r["inference_time_ms_per_image"],
            "epochs_completed": r.get("epochs_completed"),
            "weights_source": r.get("weights_source"),
            "misclassified_test_images": r["misclassified"],
            "parameters": r["total_parameters"],
        }

    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = None

    return {
        "project": "AI-Based Semiconductor Wafer Defect Pattern Detection System",
        "dataset_type": "synthetic (programmatically generated; not real fab data)",
        "dataset_size": eda["total_samples"] if eda else None,
        "class_counts": eda["class_counts"] if eda else None,
        "split_counts": split["split"].value_counts().to_dict() if split is not None else None,
        "models": models,
        "best_model": best,
        "environment": {"python": platform.python_version(), "tensorflow": tf_version,
                        "platform": platform.platform(), "processor": platform.processor() or None},
        "seed": C.SEED,
    }


def main() -> int:
    s = build_summary()
    save_json(s, C.REPORT_DIR / "final_project_summary.json")
    print("Final project summary")
    print("---------------------")
    print(f"Dataset: {s['dataset_size']} images ({s['dataset_type']})")
    print(f"Split  : {s['split_counts']}")
    for name, m in s["models"].items():
        print(f"{name:15s} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"train={m['training_time_seconds']}s  infer={m['inference_time_ms_per_image']}ms")
    if s["best_model"]:
        print(f"Best model: {s['best_model']['model_name']}")
    print("Saved: results/reports/final_project_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
