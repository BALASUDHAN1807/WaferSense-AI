"""Prediction engine.

Loads the best model recorded in results/reports/best_model.json (it never
assumes which model is best), preprocesses a wafer-map image exactly like the
training data and returns:

    predicted_class, confidence, confidence_interpretation,
    all_class_probabilities (sorted, all 9 classes)

Confidence is the model's softmax probability. It is NOT a guarantee that the
physical wafer really has that defect.

Usage:
    python src/predict.py path/to/wafer.png
    python src/predict.py path/to/wafer.png --model CNN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import config as C
from preprocessing import preprocess_image
from utils import load_json, save_json

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def interpret_confidence(confidence: float) -> str:
    for threshold, label in C.CONFIDENCE_LEVELS:
        if confidence >= threshold:
            return label
    return C.CONFIDENCE_LEVELS[-1][1]


def best_model_info() -> dict:
    if not C.BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            "best_model.json not found. Run training, then: python run_project.py evaluate "
            "and python run_project.py compare")
    return load_json(C.BEST_MODEL_FILE)


class WaferPredictor:
    """Loads a trained model once and predicts many images."""

    def __init__(self, model_name: str | None = None):
        import keras
        import models  # noqa: F401  (registers custom layer)

        if model_name is None:
            info = best_model_info()
            model_name = info["model_name"]
            self.selection = info
        else:
            if model_name not in C.MODELS:
                raise ValueError(f"Unknown model {model_name}; choose from {list(C.MODELS)}")
            self.selection = None
        path = C.MODEL_DIR / C.MODELS[model_name]["file"]
        if not path.exists():
            raise FileNotFoundError(f"Model file {path.name} not found in models/. Train it first.")
        self.model_name = model_name
        self.model = keras.models.load_model(path)

    @staticmethod
    def _open(image):
        """Validate and open a path / PIL image / array."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format '{path.suffix}'. "
                                 f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
            try:
                with Image.open(path) as im:
                    im.load()
                    return im.copy()
            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"Cannot open image {path.name}: {e}") from e
        if image is None:
            raise ValueError("No image provided")
        return image

    def predict(self, image) -> dict:
        img = self._open(image)
        if isinstance(img, Image.Image) and (img.size[0] < 8 or img.size[1] < 8):
            raise ValueError(f"Image too small ({img.size[0]}x{img.size[1]}); need at least 8x8 pixels")
        x = preprocess_image(img)[None, ...]
        probs = self.model.predict(x, verbose=0)[0].astype(float)
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        conf = float(probs[top])
        return {
            "predicted_class": C.CLASS_NAMES[top],
            "confidence": conf,
            "confidence_percent": round(conf * 100, 2),
            "confidence_interpretation": interpret_confidence(conf),
            "all_class_probabilities": {C.CLASS_NAMES[i]: round(float(probs[i]), 6) for i in order},
            "model": self.model_name,
            "note": "Confidence is the model's probability, not a guarantee of the physical defect.",
        }


_PREDICTOR = None


def predict_wafer(image, model_name: str | None = None) -> dict:
    """Functional API with a cached predictor (model loaded only once)."""
    global _PREDICTOR
    if _PREDICTOR is None or (model_name and _PREDICTOR.model_name != model_name):
        _PREDICTOR = WaferPredictor(model_name)
    return _PREDICTOR.predict(image)


def print_prediction(r: dict) -> None:
    print("\nPrediction")
    print("----------")
    print(f"Model                     : {r['model']}")
    print(f"Defect class              : {r['predicted_class']}")
    print(f"Confidence                : {r['confidence_percent']:.2f}%")
    print(f"Confidence interpretation : {r['confidence_interpretation']}")
    print("\nClass probabilities:")
    for name, p in r["all_class_probabilities"].items():
        print(f"  {name:10s} {p * 100:6.2f}%  {'#' * int(round(p * 30))}")
    print(f"\n{r['note']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predict the defect pattern of a wafer-map image")
    p.add_argument("image", help="path to a wafer-map image")
    p.add_argument("--model", default=None, help="CNN, ResNet50 or EfficientNetB0 (default: best)")
    a = p.parse_args(argv)
    try:
        result = predict_wafer(a.image, a.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    print_prediction(result)
    save_json({"image": str(a.image), **result},
              C.PREDICTION_DIR / f"prediction_{Path(a.image).stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
