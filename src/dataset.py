"""Dataset management for the Semiconductor Wafer AI project.

* Generates the synthetic wafer-map dataset (900 PNG images, 9 classes, 100 per class).
* Loads dataset/labels.csv, detecting the image-path and label columns.
* Loads images as 64x64 grayscale arrays.

The dataset is SYNTHETIC: it is generated programmatically to mimic the spatial
defect patterns found in wafer maps (same class names as the public WM-811K
dataset). It is NOT real semiconductor-fab data.

Usage:
    python src/dataset.py --generate          # create dataset/images + labels.csv
    python src/dataset.py                     # print a short summary
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from PIL import Image

import config as C

# Internal die states used while drawing a wafer (converted to pixels when saving)
OUTSIDE, GOOD, FAIL = 0, 1, 2

IMAGE_COLUMN_CANDIDATES = ["image", "image_path", "path", "filepath", "file", "filename", "img"]
LABEL_COLUMN_CANDIDATES = ["label", "class", "defect", "defect_type", "category", "target"]


# ----------------------------------------------------------- synthetic data
def _grid(size: int):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return yy, xx


def _blank_wafer(rng: np.random.Generator, size: int):
    """Return (wafer, radius_map, angle_map, R, inside_mask)."""
    c = (size - 1) / 2.0
    R = rng.uniform(0.40, 0.49) * size
    yy, xx = _grid(size)
    dy, dx = yy - c, xx - c
    r = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)
    inside = r <= R
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[inside] = GOOD
    # Background noise: every class has a few random failures, like real wafers.
    noise = rng.uniform(0.0, 0.03)
    wafer[inside & (rng.random((size, size)) < noise)] = FAIL
    return wafer, r, ang, R, inside, (yy, xx, c)


def _paint(wafer, region, inside, rng, density_low=0.65, density_high=1.0):
    density = rng.uniform(density_low, density_high)
    hit = region & inside & (rng.random(wafer.shape) < density)
    wafer[hit] = FAIL


def _blob(yy, xx, cy, cx, ry, rx, theta):
    """Rotated ellipse mask."""
    ct, st = np.cos(theta), np.sin(theta)
    y, x = yy - cy, xx - cx
    u = (x * ct + y * st) / rx
    v = (-x * st + y * ct) / ry
    return u ** 2 + v ** 2 <= 1.0


def generate_one(label: str, rng: np.random.Generator, size: int = C.IMG_SIZE) -> np.ndarray:
    wafer, r, ang, R, inside, (yy, xx, c) = _blank_wafer(rng, size)

    if label == "Center":
        cy, cx = c + rng.normal(0, 0.04 * R), c + rng.normal(0, 0.04 * R)
        rad = rng.uniform(0.15, 0.35) * R
        region = _blob(yy, xx, cy, cx, rad * rng.uniform(0.8, 1.2), rad, rng.uniform(0, np.pi))
        _paint(wafer, region, inside, rng)

    elif label == "Donut":
        inner = rng.uniform(0.25, 0.50) * R
        width = rng.uniform(0.12, 0.25) * R
        region = (r >= inner) & (r <= inner + width)
        # Sometimes only part of the ring is present.
        if rng.random() < 0.3:
            start = rng.uniform(-np.pi, np.pi)
            span = rng.uniform(1.4 * np.pi, 2 * np.pi)
            region &= ((ang - start) % (2 * np.pi)) < span
        _paint(wafer, region, inside, rng)

    elif label == "Edge-Ring":
        width = rng.uniform(0.06, 0.15) * R
        region = r >= R - width
        if rng.random() < 0.4:
            start = rng.uniform(-np.pi, np.pi)
            span = rng.uniform(1.2 * np.pi, 2 * np.pi)
            region &= ((ang - start) % (2 * np.pi)) < span
        _paint(wafer, region, inside, rng)

    elif label == "Edge-Loc":
        theta = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(0.85, 1.0) * R
        cy, cx = c + dist * np.sin(theta), c + dist * np.cos(theta)
        ry, rx = rng.uniform(0.12, 0.28) * R, rng.uniform(0.08, 0.18) * R
        region = _blob(yy, xx, cy, cx, ry, rx, theta + np.pi / 2)
        _paint(wafer, region, inside, rng)

    elif label == "Loc":
        theta = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(0.30, 0.70) * R
        cy, cx = c + dist * np.sin(theta), c + dist * np.cos(theta)
        rad = rng.uniform(0.08, 0.18) * R
        region = _blob(yy, xx, cy, cx, rad * rng.uniform(0.7, 1.4), rad, rng.uniform(0, np.pi))
        _paint(wafer, region, inside, rng)

    elif label == "Near-Full":
        _paint(wafer, inside, inside, rng, 0.70, 0.97)

    elif label == "Random":
        _paint(wafer, inside, inside, rng, 0.08, 0.30)

    elif label == "Scratch":
        n_lines = 1 if rng.random() < 0.8 else 2
        for _ in range(n_lines):
            theta = rng.uniform(0, np.pi)
            length = rng.uniform(0.6, 1.7) * R
            # Line centre somewhere inside the wafer.
            off_r, off_a = rng.uniform(0, 0.5) * R, rng.uniform(-np.pi, np.pi)
            cy, cx = c + off_r * np.sin(off_a), c + off_r * np.cos(off_a)
            curve = rng.uniform(-0.01, 0.01)  # slight curvature
            t = np.linspace(-length / 2, length / 2, int(length * 3))
            ys = cy + t * np.sin(theta) + curve * t ** 2 * np.cos(theta)
            xs = cx + t * np.cos(theta) - curve * t ** 2 * np.sin(theta)
            width = rng.choice([1, 1, 2])
            region = np.zeros_like(inside)
            for y, x in zip(ys, xs):
                y0, x0 = int(round(y)), int(round(x))
                region[max(0, y0 - width // 2): y0 + width - width // 2 + 0,
                       max(0, x0 - width // 2): x0 + width - width // 2 + 0] = True
            _paint(wafer, region, inside, rng, 0.80, 1.0)

    elif label == "None":
        pass  # background noise only

    else:
        raise ValueError(f"Unknown class: {label}")

    return wafer



def wafer_to_image(wafer: np.ndarray) -> Image.Image:
    """Encode an OUTSIDE/GOOD/FAIL map as grayscale pixels 0 / 127 / 255."""
    lut = np.array([C.PIXEL_OUTSIDE, C.PIXEL_GOOD, C.PIXEL_DEFECT], dtype=np.uint8)
    return Image.fromarray(lut[wafer], mode="L")


def generate_synthetic_dataset(per_class: int = C.SYNTHETIC_PER_CLASS, seed: int = C.SEED,
                               size: int = C.IMG_SIZE, overwrite: bool = False) -> pd.DataFrame:
    """Create dataset/images/<Class>/*.png and dataset/labels.csv."""
    if C.LABEL_FILE.exists() and not overwrite:
        raise FileExistsError(f"{C.LABEL_FILE} already exists. Use --overwrite to regenerate.")
    rng = np.random.default_rng(seed)
    rows = []
    for label in C.CLASS_NAMES:
        (C.IMAGE_DIR / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            rel_path = f"{label}/{label.lower()}_{i:04d}.png"
            wafer_to_image(generate_one(label, rng, size)).save(C.IMAGE_DIR / rel_path)
            rows.append({"image": rel_path, "label": label})
    df = pd.DataFrame(rows)
    C.LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.LABEL_FILE, index=False)
    return df


# --------------------------------------------------------------- loading

def _find_column(columns, candidates, kind):
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        f"Could not find the {kind} column in labels.csv. Found columns: {list(columns)}. "
        f"Expected one of: {candidates}")


def read_labels(label_file=C.LABEL_FILE) -> pd.DataFrame:
    """Read labels.csv and return a DataFrame with standard columns 'image' and 'label'.

    keep_default_na=False is essential: otherwise pandas turns the class name
    "None" into a missing value.
    """
    if not label_file.exists():
        raise FileNotFoundError(
            f"{label_file} not found. Create the dataset with:  python run_project.py generate")
    df = pd.read_csv(label_file, keep_default_na=False, dtype=str)
    img_col = _find_column(df.columns, IMAGE_COLUMN_CANDIDATES, "image-path")
    lab_col = _find_column(df.columns, LABEL_COLUMN_CANDIDATES, "label")
    df = df.rename(columns={img_col: "image", lab_col: "label"})[["image", "label"]]
    df["image"] = df["image"].str.strip()
    df["label"] = df["label"].str.strip()
    return df


def resolve_path(image: str):
    """Image paths in labels.csv may be relative to dataset/images or absolute."""
    from pathlib import Path
    p = Path(image)
    return p if p.is_absolute() else C.IMAGE_DIR / p


def summary() -> None:
    df = read_labels()
    print(f"Records: {len(df)}")
    print(df["label"].value_counts().reindex(C.CLASS_NAMES, fill_value=0).to_string())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Synthetic wafer dataset tools")
    p.add_argument("--generate", action="store_true", help="generate the synthetic dataset")
    p.add_argument("--per-class", type=int, default=C.SYNTHETIC_PER_CLASS)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.generate:
        df = generate_synthetic_dataset(a.per_class, a.seed, overwrite=a.overwrite)
        print(f"Generated {len(df)} synthetic wafer maps in {C.IMAGE_DIR}")
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
