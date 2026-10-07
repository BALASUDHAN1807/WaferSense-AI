"""Preprocessing pipeline shared by training, evaluation and prediction.

Steps
-----
1. Load image, convert to grayscale.
2. Resize to 64x64 (nearest neighbour, so die values are not blended).
3. Normalise pixels to [0, 1] (divide by 255).
4. Encode labels with the fixed mapping in config.CLASS_NAMES.
5. Stratified 70/15/15 train/validation/test split (seed 42), saved to
   results/reports/data_split.csv so every model uses the same samples.
6. Optional augmentation (training set only): 90/180/270-degree rotations and
   flips.  A wafer is circular, so rotating or mirroring a wafer map does not
   change its defect class (a ring is still a ring, a scratch is still a
   scratch, an edge cluster stays at the edge).

No statistics are computed from validation/test data, and nothing except the
training set is ever augmented.

Usage:
    python src/preprocessing.py      # create/verify the split and print counts
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import config as C
from dataset import read_labels, resolve_path
from utils import ensure_dirs


# ------------------------------------------------------------ single image

def preprocess_image(image) -> np.ndarray:
    """Image path / PIL image / numpy array -> float32 array (64, 64, 1) in [0, 1]."""
    if isinstance(image, Image.Image):
        im = image
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
    else:
        im = Image.open(image)
    if im.size[0] == 0 or im.size[1] == 0:
        raise ValueError("Image has invalid (zero) dimensions")
    im = im.convert("L").resize((C.IMG_SIZE, C.IMG_SIZE), Image.NEAREST)
    x = np.asarray(im, dtype=np.float32) / 255.0
    return x[..., None]


def encode_labels(labels) -> np.ndarray:
    unknown = sorted(set(labels) - set(C.CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return np.array([C.CLASS_TO_ID[l] for l in labels], dtype=np.int64)


# ------------------------------------------------------------------ split

def create_split(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    idx = np.arange(len(df))
    y = df["label"].to_numpy()
    train_idx, temp_idx = train_test_split(idx, test_size=C.VAL_RATIO + C.TEST_RATIO,
                                           stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx,
                                         test_size=C.TEST_RATIO / (C.VAL_RATIO + C.TEST_RATIO),
                                         stratify=y[temp_idx], random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    out = df[["image", "label"]].copy()
    out["split"] = split
    ensure_dirs()
    out.to_csv(C.SPLIT_FILE, index=False)
    return out


def get_split(recreate: bool = False) -> pd.DataFrame:
    """Load the saved split, creating it if missing or out of date."""
    df = read_labels()
    if not recreate and C.SPLIT_FILE.exists():
        saved = pd.read_csv(C.SPLIT_FILE, keep_default_na=False, dtype=str)
        same = (len(saved) == len(df) and set(saved["image"]) == set(df["image"]))
        if same:
            return saved
        print("Dataset changed since the split was saved -> recreating split.")
    return create_split(df)


def load_images(paths) -> np.ndarray:
    return np.stack([preprocess_image(resolve_path(p)) for p in paths]).astype(np.float32)


def load_data(recreate_split: bool = False):
    """Return X_train, X_val, X_test, y_train, y_val, y_test (always the same split)."""
    split = get_split(recreate_split)
    out = []
    for name in ("train", "val", "test"):
        part = split[split["split"] == name]
        out.append((load_images(part["image"]), encode_labels(part["label"])))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = out
    check_data(X_train, X_val, X_test, y_train, y_val, y_test, split)
    return X_train, X_val, X_test, y_train, y_val, y_test


def check_data(X_train, X_val, X_test, y_train, y_val, y_test, split=None) -> None:
    for name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        assert X.shape[1:] == (C.IMG_SIZE, C.IMG_SIZE, 1), f"{name}: bad shape {X.shape}"
        assert len(X) == len(y), f"{name}: X/y length mismatch"
        assert 0.0 <= X.min() and X.max() <= 1.0, f"{name}: pixels not in [0, 1]"
        assert y.min() >= 0 and y.max() < C.NUM_CLASSES, f"{name}: label out of range"
        assert len(np.unique(y)) == C.NUM_CLASSES, f"{name}: not all classes present"
    if split is not None:
        sets = [set(split.loc[split["split"] == s, "image"]) for s in ("train", "val", "test")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "split overlap!"


# ------------------------------------------------------------ augmentation

def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random 90-degree rotation + random flip for every image in a batch."""
    out = np.empty_like(X)
    k = rng.integers(0, 4, size=len(X))
    flip = rng.random(len(X)) < 0.5
    for i in range(len(X)):
        img = np.rot90(X[i], k[i], axes=(0, 1))
        out[i] = img[:, ::-1] if flip[i] else img
    return out


def training_dataset(X, y, batch_size: int, augment: bool = C.USE_AUGMENTATION, seed: int = C.SEED):
    """tf.data pipeline: shuffle + (optional) augmentation, training data only."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=seed,
                                                           reshuffle_each_iteration=True)
    if augment:
        def _aug(img, label):
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            img = tf.image.random_flip_left_right(img)
            return img, label
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> int:
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    split = get_split()
    print("\nPreprocessing")
    print("-------------")
    print(f"Training   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Testing    : {X_test.shape}")
    print("\nSamples per class in each split:")
    print(pd.crosstab(split["label"], split["split"]).reindex(C.CLASS_NAMES)[["train", "val", "test"]]
          .to_string())
    print(f"\nSplit saved to {C.SPLIT_FILE.relative_to(C.BASE_DIR)}")
    print("Checks passed: shapes, value range, labels, all classes present, no split overlap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
