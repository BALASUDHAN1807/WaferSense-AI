import numpy as np
import pandas as pd
import pytest

import config as C
import dataset


def test_class_mapping_is_fixed_and_complete():
    assert C.NUM_CLASSES == 9
    assert C.CLASS_NAMES == ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
                             "Near-Full", "Random", "Scratch", "None"]
    assert [C.CLASS_TO_ID[n] for n in C.CLASS_NAMES] == list(range(9))


@pytest.mark.parametrize("label", C.CLASS_NAMES)
def test_generator_produces_valid_wafer(label):
    rng = np.random.default_rng(0)
    w = dataset.generate_one(label, rng)
    assert w.shape == (C.IMG_SIZE, C.IMG_SIZE)
    assert set(np.unique(w)) <= {dataset.OUTSIDE, dataset.GOOD, dataset.FAIL}
    assert (w == dataset.OUTSIDE).any() and (w == dataset.GOOD).any()


def test_generator_is_reproducible():
    a = dataset.generate_one("Scratch", np.random.default_rng(1))
    b = dataset.generate_one("Scratch", np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_near_full_has_more_defects_than_none():
    rng = np.random.default_rng(3)
    near = np.mean([(dataset.generate_one("Near-Full", rng) == dataset.FAIL).mean() for _ in range(5)])
    none = np.mean([(dataset.generate_one("None", rng) == dataset.FAIL).mean() for _ in range(5)])
    assert near > 10 * none


def test_read_labels_keeps_none_class():
    """pandas would turn the label 'None' into NaN without keep_default_na=False."""
    df = dataset.read_labels()
    assert df["label"].isna().sum() == 0
    assert (df["label"] == "None").sum() > 0
    assert set(df["label"]) == set(C.CLASS_NAMES)


def test_read_labels_detects_alternative_column_names(tmp_path):
    f = tmp_path / "labels.csv"
    pd.DataFrame({"image_path": ["a.png"], "class": ["None"]}).to_csv(f, index=False)
    df = dataset.read_labels(f)
    assert list(df.columns) == ["image", "label"] and df["label"][0] == "None"


def test_read_labels_rejects_unknown_columns(tmp_path):
    f = tmp_path / "labels.csv"
    pd.DataFrame({"foo": ["a.png"], "bar": ["None"]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        dataset.read_labels(f)


def test_dataset_validation_passes():
    import validate_dataset
    r = validate_dataset.validate()
    assert r["passed"], r
    assert r["missing_images"] == 0 and r["unreadable_images"] == 0
    assert r["invalid_labels"] == [] and r["missing_classes"] == []
