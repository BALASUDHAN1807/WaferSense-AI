"""Shared helpers: seeding, JSON I/O, directory setup, plotting, timing."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import config as C


def ensure_dirs() -> None:
    for d in C.OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = C.SEED) -> None:
    """Make NumPy, Python and TensorFlow runs reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _to_builtin(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin), encoding="utf-8")
    return path


def load_json(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    """Path relative to the project root, with forward slashes (portable)."""
    try:
        return Path(path).resolve().relative_to(C.BASE_DIR).as_posix()
    except ValueError:
        return str(path)


@contextmanager
def timer():
    t = {"start": time.perf_counter()}
    yield t
    t["seconds"] = time.perf_counter() - t["start"]


def train_model(name: str, use_pretrained: bool = True, epochs: int | None = None) -> dict:
    """Train one model on the fixed split and save model, history, metadata and curves.

    Used by train_cnn.py, train_resnet.py and train_efficientnet.py.
    The test set is NOT touched here.
    """
    import keras

    from models import BUILDERS
    from preprocessing import load_data, training_dataset

    ensure_dirs()
    set_seed(C.SEED)
    cfg = C.MODELS[name]
    epochs = epochs or cfg["epochs"]
    X_train, X_val, _, y_train, y_val, _ = load_data()

    model = BUILDERS[name]() if name == "CNN" else BUILDERS[name](use_pretrained=use_pretrained)
    model_path = C.MODEL_DIR / cfg["file"]
    print(f"\nTraining {name}  |  weights: {model.weights_source}  |  params: {model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOPPING_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=C.REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                        save_best_only=True, verbose=0),
    ]
    train_ds = training_dataset(X_train, y_train, cfg["batch_size"])

    with timer() as t:
        hist = model.fit(train_ds, validation_data=(X_val, y_val), epochs=epochs,
                         callbacks=callbacks, verbose=2)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    # Keep the best (restored) weights as the final saved model.
    model.save(model_path)

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    meta = {
        "model": name,
        "model_path": rel(model_path),
        "weights_source": model.weights_source,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
        "training_time_seconds": round(t["seconds"], 2),
        "epochs_configured": epochs,
        "epochs_completed": len(history["loss"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": max(history["val_accuracy"]),
        "val_accuracy_at_best_epoch": history["val_accuracy"][best_epoch - 1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "best_val_loss": min(history["val_loss"]),
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "augmentation": "random 90-degree rotation + horizontal flip (train only)"
                        if C.USE_AUGMENTATION else "none",
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
    }
    save_json(history, C.REPORT_DIR / f"{cfg['slug']}_history.json")
    save_json(meta, C.REPORT_DIR / f"{cfg['slug']}_training.json")
    plot_history(history, name, cfg["slug"])

    print(f"\n{name} training finished")
    print(f"  Training time        : {meta['training_time_seconds']} s")
    print(f"  Epochs completed     : {meta['epochs_completed']} (best epoch {best_epoch})")
    print(f"  Val acc @ best epoch : {meta['val_accuracy_at_best_epoch']:.4f}")
    print(f"  Final val accuracy   : {meta['final_val_accuracy']:.4f}")
    print(f"  Saved model          : {meta['model_path']}")
    return meta


def plot_history(history: dict, model_name: str, slug: str) -> list[Path]:
    """Save training/validation accuracy and loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    epochs = range(1, len(history["loss"]) + 1)
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history[metric], marker="o", label=f"Training {title}")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label=f"Validation {title}")
        ax.set_title(f"{model_name} Training and Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = C.GRAPH_DIR / f"{slug}_{metric}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out.append(path)
    return out
