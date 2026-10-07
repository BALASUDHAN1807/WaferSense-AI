import numpy as np
import pytest
from PIL import Image

import config as C
import preprocessing as P


def test_preprocess_image_shape_and_range(tmp_path):
    img = Image.fromarray(np.random.default_rng(0).choice([0, 127, 255], size=(100, 80)).astype(np.uint8))
    path = tmp_path / "w.png"
    img.save(path)
    x = P.preprocess_image(path)
    assert x.shape == (64, 64, 1)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_preprocess_rgb_and_array_inputs_match():
    gray = np.random.default_rng(1).choice([0, 127, 255], size=(64, 64)).astype(np.uint8)
    rgb = Image.fromarray(np.stack([gray] * 3, axis=-1))
    a = P.preprocess_image(Image.fromarray(gray))
    b = P.preprocess_image(rgb)
    c = P.preprocess_image(gray)
    assert np.allclose(a, b) and np.allclose(a, c)


def test_encode_labels():
    assert P.encode_labels(["Center", "None"]).tolist() == [0, 8]
    with pytest.raises(ValueError):
        P.encode_labels(["Unknown"])


def test_split_is_stratified_and_disjoint():
    split = P.get_split()
    counts = split["split"].value_counts()
    n = len(split)
    assert counts["train"] == round(n * C.TRAIN_RATIO)
    assert counts["val"] + counts["test"] == n - counts["train"]
    sets = {s: set(split.loc[split["split"] == s, "image"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    # every class in every split, in equal proportion for a balanced dataset
    table = split.groupby(["split", "label"]).size().unstack()
    assert table.shape[1] == C.NUM_CLASSES and table.notna().all().all()


def test_split_is_reproducible():
    from dataset import read_labels
    a = P.create_split(read_labels())
    b = P.create_split(read_labels())
    assert a.equals(b)


def test_load_data_shapes():
    X_train, X_val, X_test, y_train, y_val, y_test = P.load_data()
    assert X_train.shape[1:] == (64, 64, 1)
    assert len(X_train) + len(X_val) + len(X_test) == len(P.get_split())
    assert set(np.unique(y_test)) == set(range(C.NUM_CLASSES))


def test_augmentation_keeps_shape_and_values():
    X = np.random.default_rng(2).choice([0.0, 127 / 255, 1.0], size=(4, 64, 64, 1)).astype(np.float32)
    A = P.augment_batch(X, np.random.default_rng(0))
    assert A.shape == X.shape
    # rotations/flips only rearrange pixels
    for a, x in zip(A, X):
        assert np.isclose(a.sum(), x.sum())
