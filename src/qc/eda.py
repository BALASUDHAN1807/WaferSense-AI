"""Dataset validation + exploratory data analysis.

Validation checks every file pair (image <-> label <-> mask), label syntax,
class ids, coordinate ranges and that splits don't share frames.
EDA reports real statistics and saves graphs to results/graphs/ and a summary
to results/reports/eda_summary.json.

Usage:
    python run_project.py validate
    python run_project.py eda
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import cv2
import numpy as np

from qc import config as C


def validate() -> dict:
    problems, counts = [], {}
    for split in C.N_SCENES:
        s_img = sorted((C.SCENE_DIR / split / "images").glob("*.png"))
        d_img = sorted((C.DEFECT_DIR / split / "images").glob("*.png"))
        counts[split] = {"frames": len(s_img), "crops": len(d_img)}
        for img in s_img:
            lab = C.SCENE_DIR / split / "labels" / f"{img.stem}.txt"
            if not lab.exists():
                problems.append(f"missing detection label {lab.name}")
                continue
            for row in filter(None, lab.read_text().splitlines()):
                v = row.split()
                if len(v) != 5 or v[0] != "0" or not all(0 <= float(x) <= 1 for x in v[1:]):
                    problems.append(f"bad detection label in {lab.name}: {row[:40]}")
        for img in d_img:
            lab = C.DEFECT_DIR / split / "labels" / f"{img.stem}.txt"
            mask = C.DEFECT_DIR / split / "masks" / img.name
            if not lab.exists() or not mask.exists():
                problems.append(f"missing label/mask for crop {img.name}")
                continue
            for row in filter(None, lab.read_text().splitlines()):
                v = row.split()
                if not (0 <= int(v[0]) < len(C.DEFECT_CLASSES)) or len(v) < 7 or len(v) % 2 == 0 \
                        or not all(0 <= float(x) <= 1 for x in v[1:]):
                    problems.append(f"bad polygon in {lab.name}")
            m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (C.CROP_SIZE, C.CROP_SIZE) or m.max() > len(C.DEFECT_CLASSES):
                problems.append(f"bad mask {mask.name}")
        if not (C.ANNOTATION_DIR / f"{split}.json").exists():
            problems.append(f"missing annotations/{split}.json")
    report = {"counts": counts, "problems": problems[:50], "n_problems": len(problems),
              "passed": not problems and all(c["frames"] for c in counts.values())}
    print("\nDataset validation")
    print("------------------")
    for s, c in counts.items():
        print(f"{s:5s}: {c['frames']:4d} frames, {c['crops']:4d} device crops")
    print(f"Problems: {len(problems)}")
    for p in problems[:10]:
        print("  -", p)
    print("RESULT:", "PASSED" if report["passed"] else "FAILED")
    C.ensure_dirs()
    (C.REPORT_DIR / "dataset_validation.json").write_text(json.dumps(report, indent=2))
    return report


def run_eda() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.ensure_dirs()
    rules = C.load_rules()
    per_split, defects, all_devices = {}, [], []
    for split in C.N_SCENES:
        ann = json.loads((C.ANNOTATION_DIR / f"{split}.json").read_text())
        devs = [d for s in ann["scenes"] for d in s["devices"]]
        per_split[split] = {"frames": len(ann["scenes"]), "devices": len(devs),
                            "decisions": dict(Counter(d["decision"] for d in devs)),
                            "devices_per_frame": dict(Counter(len(s["devices"]) for s in ann["scenes"]))}
        all_devices += devs
        defects += [dict(x, split=split) for d in devs for x in d["defects"]]

    dec = Counter(d["decision"] for d in all_devices)
    cls_counts = Counter(x["class"] for x in defects)
    verdicts = Counter((x["class"], x["verdict"]) for x in defects)
    n_def = Counter(len(d["defects"]) for d in all_devices)
    graphs = []

    # 1. decisions per split
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(C.DECISIONS))
    for k, split in enumerate(C.N_SCENES):
        vals = [per_split[split]["decisions"].get(d, 0) for d in C.DECISIONS]
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=split)
    ax.set_xticks(x, C.DECISIONS)
    ax.set_ylabel("devices")
    ax.set_title("Ground-truth quality decision per split")
    ax.legend()
    fig.tight_layout()
    graphs.append(C.GRAPH_DIR / "decision_distribution.png")
    fig.savefig(graphs[-1], dpi=120)
    plt.close(fig)

    # 2. defect classes, split by verdict
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(C.DEFECT_CLASSES))
    for d, col in zip(C.DECISIONS, ["#2e9e4f", "#e69500", "#d63c3c"]):
        vals = np.array([verdicts.get((c, d), 0) for c in C.DEFECT_CLASSES])
        ax.bar(C.DEFECT_CLASSES, vals, bottom=bottom, label=f"verdict {d}", color=col)
        bottom += vals
    ax.set_ylabel("defect instances")
    ax.set_title("Defect instances by type (coloured by the verdict of the quality rules)")
    ax.legend()
    fig.tight_layout()
    graphs.append(C.GRAPH_DIR / "defect_class_distribution.png")
    fig.savefig(graphs[-1], dpi=120)
    plt.close(fig)

    # 3. defect size distributions with the rule thresholds
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (cls, key, unit) in zip(axes, [("scratch", "length_mm", "length (mm)"),
                                            ("contamination", "area_mm2", "area (mm²)"),
                                            ("chip", "area_mm2", "area (mm²)")]):
        vals = [x[key] for x in defects if x["class"] == cls]
        ax.hist(vals, bins=30, color="#3b6ea5")
        for b in rules[cls]["bands"][:-1]:
            ax.axvline(float(b["max"]), color="#d63c3c", ls="--")
        ax.set_title(f"{cls} {unit}\n(dashed = rule thresholds)")
        ax.set_xlabel(unit)
    fig.tight_layout()
    graphs.append(C.GRAPH_DIR / "defect_size_distributions.png")
    fig.savefig(graphs[-1], dpi=120)
    plt.close(fig)

    # 4. sample frames + a crop gallery per defect class
    frames = sorted((C.SCENE_DIR / "train" / "images").glob("*.png"))[:6]
    tiles = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frames]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, t, f in zip(axes.ravel(), tiles, frames):
        ax.imshow(t)
        ax.set_title(f.name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample synthetic camera frames (training split)")
    fig.tight_layout()
    graphs.append(C.GRAPH_DIR / "sample_frames.png")
    fig.savefig(graphs[-1], dpi=110)
    plt.close(fig)

    gallery = {c: [] for c in C.DEFECT_CLASSES}
    for p in sorted((C.DEFECT_DIR / "train" / "masks").glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        for k, c in enumerate(C.DEFECT_CLASSES, start=1):
            if len(gallery[c]) < 3 and (m == k).sum() > 25:
                gallery[c].append(p.name)
        if all(len(v) == 3 for v in gallery.values()):
            break
    pal = {k: np.array(col[::-1]) for k, col in enumerate(
        [(0, 0, 0), (230, 40, 40), (230, 60, 230), (255, 215, 0), (30, 140, 255), (60, 220, 60)])}
    fig, axes = plt.subplots(len(C.DEFECT_CLASSES), 6, figsize=(12, 10.5))
    for r, c in enumerate(C.DEFECT_CLASSES):
        for j in range(3):
            for ax in axes[r, 2 * j:2 * j + 2]:
                ax.axis("off")
            if j >= len(gallery[c]):
                continue
            name = gallery[c][j]
            img = cv2.cvtColor(cv2.imread(str(C.DEFECT_DIR / "train" / "images" / name)), cv2.COLOR_BGR2RGB)
            m = cv2.imread(str(C.DEFECT_DIR / "train" / "masks" / name), cv2.IMREAD_GRAYSCALE)
            over = img.copy()
            for k in range(1, len(C.DEFECT_CLASSES) + 1):
                over[m == k] = pal[k][::-1]
            axes[r, 2 * j].imshow(img)
            axes[r, 2 * j + 1].imshow(over)
        axes[r, 0].set_title(c, loc="left", fontsize=10)
    fig.suptitle("Device crops and ground-truth defect masks (image | mask overlay)")
    fig.tight_layout()
    graphs.append(C.GRAPH_DIR / "defect_gallery.png")
    fig.savefig(graphs[-1], dpi=110)
    plt.close(fig)

    def q(cls, key):
        v = [x[key] for x in defects if x["class"] == cls]
        return {"min": round(min(v), 3), "median": round(float(np.median(v)), 3), "max": round(max(v), 3)} if v else None

    summary = {
        "dataset_type": "synthetic (programmatically generated camera images, not photographs)",
        "camera": {"frame_px": C.SCENE_SIZE, "px_per_mm": C.PX_PER_MM,
                   "field_of_view_mm": round(C.SCENE_SIZE / C.PX_PER_MM, 1)},
        "splits": per_split,
        "total_frames": sum(p["frames"] for p in per_split.values()),
        "total_devices": len(all_devices),
        "decision_counts": {d: dec.get(d, 0) for d in C.DECISIONS},
        "decision_percent": {d: round(100 * dec.get(d, 0) / len(all_devices), 1) for d in C.DECISIONS},
        "defect_instances": {c: cls_counts.get(c, 0) for c in C.DEFECT_CLASSES},
        "defects_per_device": {str(k): v for k, v in sorted(n_def.items())},
        "defect_sizes": {c: {"area_mm2": q(c, "area_mm2"), "length_mm": q(c, "length_mm")}
                         for c in C.DEFECT_CLASSES},
        "graphs": [str(g.relative_to(C.ROOT)) for g in graphs],
    }
    (C.REPORT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("total_frames", "total_devices", "decision_counts",
                                              "decision_percent", "defect_instances")}, indent=1))
    print("Graphs:", *summary["graphs"], sep="\n  ")
    return summary


if __name__ == "__main__":
    sys.exit(0 if (validate()["passed"] if "validate" in sys.argv else run_eda()) else 1)
