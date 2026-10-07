"""Prediction tests use the trained CNN (models/cnn_model.keras)."""
import numpy as np
import pytest
from PIL import Image

import config as C
import predict

pytestmark = pytest.mark.skipif(not (C.MODEL_DIR / C.MODELS["CNN"]["file"]).exists(),
                                reason="CNN not trained yet - run: python run_project.py train-cnn")


@pytest.fixture(scope="module")
def predictor():
    return predict.WaferPredictor("CNN")


@pytest.fixture(scope="module")
def test_images():
    from preprocessing import get_split
    split = get_split()
    return split[split["split"] == "test"]


def test_prediction_output_structure(predictor, test_images):
    row = test_images.iloc[0]
    r = predictor.predict(C.IMAGE_DIR / row["image"])
    assert r["predicted_class"] in C.CLASS_NAMES
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["confidence_interpretation"] in {lab for _, lab in C.CONFIDENCE_LEVELS}
    assert set(r["all_class_probabilities"]) == set(C.CLASS_NAMES)


def test_probabilities_sum_to_one_and_are_sorted(predictor, test_images):
    r = predictor.predict(C.IMAGE_DIR / test_images.iloc[1]["image"])
    probs = list(r["all_class_probabilities"].values())
    assert abs(sum(probs) - 1.0) < 1e-3
    assert probs == sorted(probs, reverse=True)
    assert r["predicted_class"] == next(iter(r["all_class_probabilities"]))


def test_predicts_most_test_images_correctly(predictor, test_images):
    sample = test_images.groupby("label").head(2)
    correct = sum(predictor.predict(C.IMAGE_DIR / p)["predicted_class"] == lab
                  for p, lab in zip(sample["image"], sample["label"]))
    assert correct / len(sample) >= 0.8


def test_pil_and_path_inputs_agree(predictor, test_images):
    path = C.IMAGE_DIR / test_images.iloc[2]["image"]
    a = predictor.predict(path)
    b = predictor.predict(Image.open(path))
    assert a["predicted_class"] == b["predicted_class"]
    assert abs(a["confidence"] - b["confidence"]) < 1e-6


def test_missing_image_raises(predictor):
    with pytest.raises(FileNotFoundError):
        predictor.predict("does/not/exist.png")


def test_invalid_image_raises(predictor, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ValueError):
        predictor.predict(bad)


def test_unsupported_format_raises(predictor, tmp_path):
    f = tmp_path / "wafer.txt"
    f.write_text("hello")
    with pytest.raises(ValueError):
        predictor.predict(f)


def test_tiny_image_raises(predictor):
    with pytest.raises(ValueError):
        predictor.predict(Image.fromarray(np.zeros((2, 2), dtype=np.uint8)))


@pytest.mark.parametrize("conf,label", [(0.95, "Very high confidence"), (0.75, "High confidence"),
                                        (0.55, "Moderate confidence"), (0.2, "Low confidence")])
def test_confidence_interpretation(conf, label):
    assert predict.interpret_confidence(conf) == label
