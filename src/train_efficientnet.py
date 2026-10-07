"""Train the EfficientNetB0 transfer-learning model on the fixed training split.

Outputs:
    models/efficientnetb0_model.keras
    results/reports/efficientnetb0_history.json, _training.json
    results/graphs/efficientnetb0_accuracy.png, _loss.png

Usage:
    python src/train_efficientnet.py [--epochs N]
"""
import argparse
import sys

from utils import train_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train EfficientNetB0")
    p.add_argument("--epochs", type=int, default=None, help="override epochs from config.py")
    p.add_argument("--no-pretrained", action="store_true", help="do not use ImageNet weights")
    a = p.parse_args(argv)
    train_model("EfficientNetB0", use_pretrained=not a.no_pretrained, epochs=a.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
