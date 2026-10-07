"""Dataset validation.

Checks labels.csv and every image: missing files, unreadable images,
duplicate records, duplicate image content, invalid labels, image shapes and
class counts.  Exits with a non-zero status if the dataset is invalid.

Usage:
    python src/validate_dataset.py
"""
from __future__ import annotations

import hashlib
import sys

from PIL import Image

import config as C
from dataset import read_labels, resolve_path
from utils import ensure_dirs, save_json


def validate() -> dict:
    df = read_labels()

    missing, unreadable, sizes, modes, hashes = [], [], {}, {}, {}
    for rel_path in df["image"]:
        path = resolve_path(rel_path)
        if not path.exists():
            missing.append(rel_path)
            continue
        try:
            with Image.open(path) as im:
                im.load()
                sizes[f"{im.size[0]}x{im.size[1]}"] = sizes.get(f"{im.size[0]}x{im.size[1]}", 0) + 1
                modes[im.mode] = modes.get(im.mode, 0) + 1
            hashes.setdefault(hashlib.md5(path.read_bytes()).hexdigest(), []).append(rel_path)
        except Exception:
            unreadable.append(rel_path)

    invalid_labels = sorted(set(df["label"]) - set(C.CLASS_NAMES))
    duplicate_records = int(df.duplicated(subset=["image"]).sum())
    duplicate_content = sum(len(v) - 1 for v in hashes.values() if len(v) > 1)
    counts = df["label"].value_counts().reindex(C.CLASS_NAMES, fill_value=0)
    missing_classes = [c for c in C.CLASS_NAMES if counts[c] == 0]

    report = {
        "total_records": len(df),
        "valid_images": len(df) - len(missing) - len(unreadable),
        "missing_images": len(missing),
        "unreadable_images": len(unreadable),
        "duplicate_records": duplicate_records,
        "duplicate_image_content": duplicate_content,
        "invalid_labels": invalid_labels,
        "missing_classes": missing_classes,
        "image_sizes": sizes,
        "image_modes": modes,
        "class_distribution": counts.to_dict(),
        "examples_missing": missing[:10],
        "examples_unreadable": unreadable[:10],
    }
    # Fundamental problems make the dataset unusable for training.
    report["passed"] = (report["valid_images"] > 0 and not missing and not unreadable
                        and not invalid_labels and not missing_classes and duplicate_records == 0)
    return report


def print_report(r: dict) -> None:
    print("\nDataset validation")
    print("------------------")
    for key in ("total_records", "valid_images", "missing_images", "unreadable_images",
                "duplicate_records", "duplicate_image_content", "invalid_labels",
                "missing_classes", "image_sizes", "image_modes"):
        print(f"{key.replace('_', ' ').capitalize():26s}: {r[key]}")
    print("\nClass distribution:")
    for name, n in r["class_distribution"].items():
        print(f"  {name:10s} {n}")
    print("\nRESULT:", "PASSED" if r["passed"] else "FAILED")


def main() -> int:
    ensure_dirs()
    try:
        report = validate()
    except (FileNotFoundError, ValueError) as e:
        print(f"Dataset validation FAILED: {e}")
        return 1
    print_report(report)
    save_json(report, C.REPORT_DIR / "dataset_validation.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
