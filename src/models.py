"""Model architectures (no training happens here).

All three models take the SAME input: a 64x64x1 wafer map scaled to [0, 1].
The two transfer-learning models convert it internally (resize to 224x224,
grayscale -> 3 channels, ImageNet-style scaling), so preprocessing outside the
model is identical for every model and the prediction engine never needs to
know which architecture it is using.

    build_cnn()            custom CNN
    build_resnet50()       ResNet50 + new 9-class head (ImageNet weights, frozen base)
    build_efficientnet()   EfficientNetB0 + new 9-class head (ImageNet weights, frozen base)

If ImageNet weights cannot be downloaded (no internet), the transfer models
fall back to random initialisation with a TRAINABLE base and record this in
`model.weights_source` / the training report.

Usage:
    python src/models.py        # build all models and check output shape (None, 9)
"""
from __future__ import annotations

import sys

import tensorflow as tf
import keras

import config as C

INPUT_SHAPE = (C.IMG_SIZE, C.IMG_SIZE, 1)


@keras.saving.register_keras_serializable(package="wafer")
class WaferToRGB(keras.layers.Layer):
    """[0,1] grayscale (H,W,1) -> resized 3-channel image in the format a
    pretrained backbone expects.

    mode="caffe" (ResNet50): 0-255 scale minus ImageNet channel means (BGR).
    mode="raw255" (EfficientNetB0): 0-255 scale; the backbone rescales itself.
    """

    def __init__(self, size: int = 224, mode: str = "raw255", **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.mode = mode

    def call(self, x):
        x = tf.image.resize(x, (self.size, self.size), method="bilinear")
        x = tf.repeat(x, 3, axis=-1) * 255.0
        if self.mode == "caffe":
            x = x - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
        return x

    def get_config(self):
        return {**super().get_config(), "size": self.size, "mode": self.mode}


def _compile(model: keras.Model, lr: float) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_cnn(lr: float = C.MODELS["CNN"]["learning_rate"]) -> keras.Model:
    """Custom CNN: 3 x (Conv2D + ReLU + MaxPool) -> Flatten -> Dense -> Dropout -> Softmax."""
    model = keras.Sequential([
        keras.layers.Input(shape=INPUT_SHAPE),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(C.NUM_CLASSES, activation="softmax"),
    ], name="CNN")
    model.weights_source = "random init (trained from scratch)"
    return _compile(model, lr)


def _load_backbone(app_fn, size: int, use_pretrained: bool):
    if use_pretrained:
        try:
            base = app_fn(include_top=False, weights="imagenet", input_shape=(size, size, 3))
            base.trainable = False
            return base, "imagenet (frozen base)"
        except Exception as e:  # no internet / download blocked
            print(f"[models] Could not load ImageNet weights ({type(e).__name__}: {e}).\n"
                  "         Falling back to random initialisation with a trainable base.")
    base = app_fn(include_top=False, weights=None, input_shape=(size, size, 3))
    base.trainable = True
    return base, "random init (trainable base, ImageNet weights unavailable)"


def _transfer_model(name: str, app_fn, mode: str, use_pretrained: bool) -> keras.Model:
    cfg = C.MODELS[name]
    size = cfg["input_size"]
    base, source = _load_backbone(app_fn, size, use_pretrained)
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = WaferToRGB(size=size, mode=mode, name="wafer_to_rgb")(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(C.NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=name)
    model.weights_source = source
    return _compile(model, cfg["learning_rate"])


def build_resnet50(use_pretrained: bool = True) -> keras.Model:
    return _transfer_model("ResNet50", keras.applications.ResNet50, "caffe", use_pretrained)


def build_efficientnet(use_pretrained: bool = True) -> keras.Model:
    return _transfer_model("EfficientNetB0", keras.applications.EfficientNetB0, "raw255",
                           use_pretrained)


BUILDERS = {
    "CNN": build_cnn,
    "ResNet50": build_resnet50,
    "EfficientNetB0": build_efficientnet,
}


def main() -> int:
    for name, builder in BUILDERS.items():
        kwargs = {} if name == "CNN" else {"use_pretrained": "--no-download" not in sys.argv}
        m = builder(**kwargs)
        trainable = sum(int(tf.size(w)) for w in m.trainable_weights)
        print(f"{name:15s} output={m.output_shape}  params={m.count_params():,}  "
              f"trainable={trainable:,}  weights={m.weights_source}")
        assert m.output_shape == (None, C.NUM_CLASSES)
    print("All models built successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
