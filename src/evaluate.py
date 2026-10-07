"""Evaluate every trained model on the held-out TEST set.

For each model: accuracy, macro precision/recall/F1, per-class
precision/recall/F1, confusion matrix, classification report, and
inference time per image (measured programmatically).

Outputs:
    results/reports/<slug>_classification_report.json
    results/confusion_matrices/<slug>_confusion_matrix.png
    results/predictions/<slug>_test_predictions.csv
    results/reports/all_model_results.json

Usage:
    python src/evaluate.py                # all models that exist in models/
    python src/evaluate.py --models CNN
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

import config as C
from utils import ensure_dirs, load_json, rel, save_json


def load_model(name: str):
    import keras
    import models  # noqa: F401  (registers the custom WaferToRGB layer)

    path = C.MODEL_DIR / C.MODELS[name]["file"]
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} not found. Train it first, e.g. "
                                f"python run_project.py train-{C.MODELS[name]['slug']}")
    return keras.models.load_model(path)


def measure_inference_time(model, X, repeats: int = 3) -> float:
    """Average seconds per image for batch prediction (after one warm-up call)."""
    model.predict(X[:8], verbose=0)  # warm-up (graph tracing)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X, batch_size=32, verbose=0)
        times.append((time.perf_counter() - t0) / len(X))
    return float(np.median(times))


def plot_confusion(cm: np.ndarray, name: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(C.NUM_CLASSES), C.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(C.NUM_CLASSES), C.CLASS_NAMES)
    thresh = cm.max() / 2
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"{name} Confusion Matrix - Held-out Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_model(name: str, X_test, y_test, test_images):
    cfg = C.MODELS[name]
    model = load_model(name)
    probs = model.predict(X_test, batch_size=32, verbose=0)
    preds = probs.argmax(axis=1)
    inference = measure_inference_time(model, X_test)

    labels = list(range(C.NUM_CLASSES))
    report = classification_report(y_test, preds, labels=labels, target_names=C.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels)

    save_json(report, C.REPORT_DIR / f"{cfg['slug']}_classification_report.json")
    cm_path = C.CM_DIR / f"{cfg['slug']}_confusion_matrix.png"
    plot_confusion(cm, name, cm_path)

    pd.DataFrame({
        "image": test_images,
        "actual": [C.CLASS_NAMES[i] for i in y_test],
        "predicted": [C.CLASS_NAMES[i] for i in preds],
        "confidence": probs.max(axis=1).round(4),
        "correct": preds == y_test,
    }).to_csv(C.PREDICTION_DIR / f"{cfg['slug']}_test_predictions.csv", index=False)

    training = {}
    train_file = C.REPORT_DIR / f"{cfg['slug']}_training.json"
    if train_file.exists():
        training = load_json(train_file)

    result = {
        "model": name,
        "model_path": rel(C.MODEL_DIR / cfg["file"]),
        "weights_source": training.get("weights_source"),
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_precision": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score")}
                      for c in C.CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
        "misclassified": int((preds != y_test).sum()),
        "inference_time_ms_per_image": round(inference * 1000, 4),
        "training_time_seconds": training.get("training_time_seconds"),
        "epochs_completed": training.get("epochs_completed"),
        "total_parameters": int(model.count_params()),
        "confusion_matrix_image": rel(cm_path),
    }
    return result, preds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate trained models on the test set")
    p.add_argument("--models", nargs="*", default=list(C.MODELS))
    a = p.parse_args(argv)

    from preprocessing import get_split, load_data
    ensure_dirs()
    _, _, X_test, _, _, y_test = load_data()
    split = get_split()
    test_images = split.loc[split["split"] == "test", "image"].to_numpy()

    results, skipped = [], []
    for name in a.models:
        if not (C.MODEL_DIR / C.MODELS[name]["file"]).exists():
            skipped.append(name)
            continue
        print(f"Evaluating {name} on {len(y_test)} test images...")
        r, preds = evaluate_model(name, X_test, y_test, test_images)
        results.append(r)
        print(classification_report(y_test, preds, labels=list(range(C.NUM_CLASSES)),
                                    target_names=C.CLASS_NAMES, digits=4, zero_division=0))

    if not results:
        print("No trained models found. Train at least one model first.")
        return 1

    save_json({"split": "test", "results": results}, C.REPORT_DIR / "all_model_results.json")

    print("\nModel           | Accuracy | Precision | Recall | Macro F1 | Inference (ms/img)")
    print("----------------+----------+-----------+--------+----------+-------------------")
    for r in results:
        print(f"{r['model']:15s} | {r['accuracy']:.4f}   | {r['macro_precision']:.4f}    | "
              f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f}   | {r['inference_time_ms_per_image']:.3f}")
    if skipped:
        print(f"\nSkipped (not trained yet): {', '.join(skipped)}")
    print(f"\nSaved: {rel(C.REPORT_DIR / 'all_model_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
