"""Exploratory Data Analysis.

Generates real statistics and graphs from the dataset:
class distribution, percentages, missing values, duplicates, image
dimensions, representative wafer maps per class, pixel-intensity statistics,
defect-pixel ratio per class and a mean wafer map per class.

Outputs go to results/graphs/ and results/reports/eda_summary.json.

Usage:
    python src/eda.py
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as C
from dataset import read_labels, resolve_path
from utils import ensure_dirs, rel, save_json


def _load_raw(df):
    imgs, dims = [], []
    for p in df["image"]:
        with Image.open(resolve_path(p)) as im:
            dims.append(im.size)
            imgs.append(np.asarray(im.convert("L"), dtype=np.uint8))
    return imgs, dims


def run_eda() -> dict:
    ensure_dirs()
    df = read_labels()
    imgs, dims = _load_raw(df)
    labels = df["label"].to_numpy()
    counts = df["label"].value_counts().reindex(C.CLASS_NAMES, fill_value=0)
    outputs = []

    # 1. class distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color="#3b6ea5")
    ax.bar_label(bars)
    ax.set_title("Class Distribution of the Synthetic Wafer Dataset")
    ax.set_xlabel("Wafer Defect Class")
    ax.set_ylabel("Number of Images")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = C.GRAPH_DIR / "class_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    outputs.append(path)

    # 2. representative wafer maps (3 per class)
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(len(C.CLASS_NAMES), 3, figsize=(6, 2 * len(C.CLASS_NAMES)))
    for r, name in enumerate(C.CLASS_NAMES):
        idx = np.flatnonzero(labels == name)
        pick = rng.choice(idx, size=min(3, len(idx)), replace=False) if len(idx) else []
        for c in range(3):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick):
                ax.imshow(imgs[pick[c]], cmap="gray", vmin=0, vmax=255)
            if c == 0:
                ax.set_title(name, loc="left", fontsize=10)
    fig.suptitle("Representative Wafer Maps (3 per class)")
    fig.tight_layout()
    path = C.GRAPH_DIR / "sample_wafer_maps.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    outputs.append(path)

    # 3. image dimensions
    widths = np.array([d[0] for d in dims])
    heights = np.array([d[1] for d in dims])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, heights, alpha=0.4)
    ax.set_title("Image Dimensions")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    fig.tight_layout()
    path = C.GRAPH_DIR / "image_dimensions.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    outputs.append(path)

    # 4. pixel intensity histogram
    all_px = np.concatenate([im.ravel() for im in imgs])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_px, bins=64, color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_title("Pixel Intensity Distribution (all images)")
    ax.set_xlabel("Pixel value (0 = outside, 127 = good die, 255 = defective die)")
    ax.set_ylabel("Count (log scale)")
    fig.tight_layout()
    path = C.GRAPH_DIR / "pixel_intensity_histogram.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    outputs.append(path)

    # 5. defect-die ratio per class (failed dies / dies on wafer)
    def defect_ratio(im):
        on_wafer = im > 64
        return float((im[on_wafer] > 191).mean()) if on_wafer.any() else 0.0
    ratios = np.array([defect_ratio(im) for im in imgs])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([ratios[labels == n] for n in C.CLASS_NAMES], tick_labels=C.CLASS_NAMES)
    ax.set_title("Defective-Die Ratio per Class")
    ax.set_ylabel("Defective dies / dies on wafer")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = C.GRAPH_DIR / "defect_ratio_per_class.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    outputs.append(path)

    # 6. mean wafer map per class (where defects usually appear)
    same_size = len({im.shape for im in imgs}) == 1
    if same_size:
        fig, axes = plt.subplots(1, len(C.CLASS_NAMES), figsize=(2 * len(C.CLASS_NAMES), 2.4))
        for ax, name in zip(axes, C.CLASS_NAMES):
            stack = np.stack([imgs[i] > 191 for i in np.flatnonzero(labels == name)])
            ax.imshow(stack.mean(axis=0), cmap="magma", vmin=0, vmax=1)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        fig.suptitle("Defect Frequency Map per Class (brighter = defects more common)")
        fig.tight_layout()
        path = C.GRAPH_DIR / "mean_defect_map_per_class.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        outputs.append(path)

    summary = {
        "total_samples": int(len(df)),
        "num_classes": int((counts > 0).sum()),
        "class_counts": counts.to_dict(),
        "class_percentages": (counts / counts.sum() * 100).round(2).to_dict(),
        "is_balanced": bool(counts.max() == counts.min()),
        "missing_values_in_labels_csv": int((df == "").sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
        "image_width": {"min": int(widths.min()), "max": int(widths.max()), "mean": float(widths.mean())},
        "image_height": {"min": int(heights.min()), "max": int(heights.max()), "mean": float(heights.mean())},
        "pixel_stats": {"min": int(all_px.min()), "max": int(all_px.max()),
                        "mean": round(float(all_px.mean()), 3), "std": round(float(all_px.std()), 3),
                        "unique_values": np.unique(all_px).tolist()[:20]},
        "defect_ratio_per_class_mean": {n: round(float(ratios[labels == n].mean()), 4) for n in C.CLASS_NAMES},
        "dataset_type": "synthetic (programmatically generated, not real fab data)",
        "graphs": [rel(p) for p in outputs],
    }
    save_json(summary, C.REPORT_DIR / "eda_summary.json")
    return summary


def main() -> int:
    s = run_eda()
    print("\nExploratory Data Analysis")
    print("-------------------------")
    print(f"Total samples : {s['total_samples']}")
    print(f"Classes       : {s['num_classes']}  (balanced: {s['is_balanced']})")
    print(f"Image size    : {s['image_width']['min']}-{s['image_width']['max']} x "
          f"{s['image_height']['min']}-{s['image_height']['max']}")
    print(f"Pixel mean/std: {s['pixel_stats']['mean']} / {s['pixel_stats']['std']}")
    print("Defective-die ratio per class (mean):")
    for k, v in s["defect_ratio_per_class_mean"].items():
        print(f"  {k:10s} {v:.3f}")
    print("Graphs saved:")
    for g in s["graphs"]:
        print("  ", g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
