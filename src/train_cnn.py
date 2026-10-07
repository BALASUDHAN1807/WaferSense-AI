"""Train the custom CNN on the fixed training split.

Outputs:
    models/cnn_model.keras
    results/reports/cnn_history.json, _training.json
    results/graphs/cnn_accuracy.png, _loss.png

Usage:
    python src/train_cnn.py [--epochs N]
"""
import argparse
import sys

from utils import train_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train CNN")
    p.add_argument("--epochs", type=int, default=None, help="override epochs from config.py")
    a = p.parse_args(argv)
    train_model("CNN", use_pretrained=True, epochs=a.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
