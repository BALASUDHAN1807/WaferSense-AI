import numpy as np
import pytest

import config as C
import models


@pytest.fixture(scope="module")
def batch():
    return np.random.default_rng(0).random((2, 64, 64, 1)).astype(np.float32)


def _check(model, batch):
    assert model.output_shape == (None, C.NUM_CLASSES)
    out = model.predict(batch, verbose=0)
    assert out.shape == (2, C.NUM_CLASSES)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)  # softmax


def test_cnn_output(batch):
    _check(models.build_cnn(), batch)


def test_resnet50_output(batch):
    _check(models.build_resnet50(use_pretrained=False), batch)


def test_efficientnet_output(batch):
    _check(models.build_efficientnet(use_pretrained=False), batch)


@pytest.mark.parametrize("mode", ["caffe", "raw255"])
def test_wafer_to_rgb_layer(mode):
    layer = models.WaferToRGB(size=96, mode=mode)
    y = layer(np.ones((1, 64, 64, 1), dtype=np.float32)).numpy()
    assert y.shape == (1, 96, 96, 3)
    if mode == "raw255":
        assert np.isclose(y.max(), 255.0)
    else:
        assert np.allclose(y[0, 0, 0], 255.0 - np.array([103.939, 116.779, 123.68]), atol=1e-3)


def test_custom_layer_survives_save_and_load(tmp_path, batch):
    import keras
    inp = keras.Input((64, 64, 1))
    out = keras.layers.GlobalAveragePooling2D()(models.WaferToRGB(32)(inp))
    m = keras.Model(inp, out)
    path = tmp_path / "m.keras"
    m.save(path)
    m2 = keras.models.load_model(path)
    assert np.allclose(m.predict(batch, verbose=0), m2.predict(batch, verbose=0))
