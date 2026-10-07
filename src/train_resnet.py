"""Train the ResNet50 transfer-learning model on the fixed training split.

Outputs:
    models/resnet50_model.keras
    results/reports/resnet50_history.json, _training.json
    results/graphs/resnet50_accuracy.png, _loss.png

Usage:
    python src/train_resnet.py [--epochs N]
"""
import argparse
import sys

from utils import train_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train ResNet50")
    p.add_argument("--epochs", type=int, default=None, help="override epochs from config.py")
    p.add_argument("--no-pretrained", action="store_true", help="do not use ImageNet weights")
    a = p.parse_args(argv)
    train_model("ResNet50", use_pretrained=not a.no_pretrained, epochs=a.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
