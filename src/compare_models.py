"""Compare all evaluated models and automatically select the best one.

Reads the ACTUAL results written by evaluate.py (nothing is hard-coded).

Selection rule:
    1. Highest macro F1-score on the test set.
    2. If models are within config.TIE_MARGIN on macro F1, prefer higher accuracy.
    3. If still tied, prefer the faster model (lower inference time).

Outputs:
    results/reports/model_comparison.csv
    results/reports/best_model.json
    results/graphs/model_comparison.png
    results/graphs/comparison_<metric>.png   (accuracy, precision, recall, f1)
    results/graphs/comparison_time.png

Usage:
    python src/compare_models.py
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from utils import load_json, rel, save_json

METRICS = [("accuracy", "Accuracy"), ("macro_precision", "Macro Precision"),
           ("macro_recall", "Macro Recall"), ("macro_f1", "Macro F1")]


def build_table() -> pd.DataFrame:
    path = C.REPORT_DIR / "all_model_results.json"
    if not path.exists():
        raise FileNotFoundError("all_model_results.json not found. Run: python run_project.py evaluate")
    results = load_json(path)["results"]
    rows = [{
        "Model": r["model"],
        "Accuracy": r["accuracy"],
        "Macro Precision": r["macro_precision"],
        "Macro Recall": r["macro_recall"],
        "Macro F1": r["macro_f1"],
        "Training Time (s)": r.get("training_time_seconds"),
        "Inference Time (ms/image)": r["inference_time_ms_per_image"],
        "Parameters": r.get("total_parameters"),
        "Weights": r.get("weights_source"),
        "Model Path": r["model_path"],
    } for r in results]
    return pd.DataFrame(rows)


def select_best(df: pd.DataFrame) -> tuple[pd.Series, str]:
    top_f1 = df["Macro F1"].max()
    tied = df[df["Macro F1"] >= top_f1 - C.TIE_MARGIN]
    if len(tied) == 1:
        best = tied.iloc[0]
        others = df[df["Model"] != best["Model"]]
        reason = (f"{best['Model']} has the highest macro F1 ({best['Macro F1']:.4f})"
                  + (f", ahead of the next model by {best['Macro F1'] - others['Macro F1'].max():.4f}."
                     if len(others) else "."))
        return best, reason
    tied = tied.sort_values(["Accuracy", "Inference Time (ms/image)"], ascending=[False, True])
    best = tied.iloc[0]
    names = ", ".join(tied["Model"])
    reason = (f"Macro F1 was effectively tied (within {C.TIE_MARGIN}) between {names}; "
              f"{best['Model']} was selected by the tie-breakers (higher accuracy "
              f"{best['Accuracy']:.4f}, then faster inference "
              f"{best['Inference Time (ms/image)']:.3f} ms/image).")
    return best, reason


def plot_comparison(df: pd.DataFrame) -> list:
    out = []
    x = np.arange(len(df))
    width = 0.8 / len(METRICS)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (key, label) in enumerate(METRICS):
        bars = ax.bar(x + (i - (len(METRICS) - 1) / 2) * width, df[label], width, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x, df["Model"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model Comparison on the Held-out Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = C.GRAPH_DIR / "model_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    out.append(path)

    for key, label in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(df["Model"], df[label], color=["#3b6ea5", "#e07b39", "#4c9a5f"][:len(df)])
        ax.bar_label(bars, fmt="%.4f")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{label} by Model (test set)")
        fig.tight_layout()
        path = C.GRAPH_DIR / f"comparison_{key.replace('macro_', '')}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["Training Time (s)", "Inference Time (ms/image)"]):
        vals = df[col].fillna(0)
        bars = ax.bar(df["Model"], vals, color="#7a6fb0")
        ax.bar_label(bars, fmt="%.2f")
        ax.set_title(col)
    fig.suptitle("Computational Cost")
    fig.tight_layout()
    path = C.GRAPH_DIR / "comparison_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    out.append(path)
    return out


def main() -> int:
    try:
        df = build_table()
    except FileNotFoundError as e:
        print(e)
        return 1
    df.to_csv(C.REPORT_DIR / "model_comparison.csv", index=False)
    graphs = plot_comparison(df)
    best, reason = select_best(df)

    best_info = {
        "model_name": best["Model"],
        "model_path": best["Model Path"],
        "selection_metric": "macro_f1",
        "metric_value": float(best["Macro F1"]),
        "accuracy": float(best["Accuracy"]),
        "inference_time_ms_per_image": float(best["Inference Time (ms/image)"]),
        "weights_source": best["Weights"],
        "tie_breakers": ["accuracy", "inference_time"],
        "tie_margin": C.TIE_MARGIN,
        "reason": reason,
        "compared_models": df["Model"].tolist(),
    }
    save_json(best_info, C.BEST_MODEL_FILE)

    pd.set_option("display.width", 160)
    print("\nModel comparison (test set)")
    print(df.drop(columns=["Model Path", "Weights"]).to_string(index=False,
                                                                float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest model: {best['Model']}")
    print(f"Reason    : {reason}")
    print(f"\nSaved: {rel(C.REPORT_DIR / 'model_comparison.csv')}, {rel(C.BEST_MODEL_FILE)}")
    for g in graphs:
        print("       ", rel(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
