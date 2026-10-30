"""PatchCore-guided crop used by the defect-type classifier (training and web demo agree)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "vercel"))

from qc.real.type_classifier import crop_around  # noqa: E402


def _engine_crop_box(W, H, amap, frac=0.4):
    from real_lite import RealEngine
    e = RealEngine.__new__(RealEngine)                       # no model files needed
    e.tc = {"crop_fraction": frac}

    class Img:
        size = (W, H)
    return e.crop_box(Img(), amap)


def test_crop_matches_training_and_stays_inside(tmp_path):
    im = np.random.default_rng(0).integers(0, 255, (300, 400, 3), np.uint8)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), im)
    for y, x in ((0, 0), (31, 31), (10, 20)):
        amap = np.zeros((32, 32), np.float32)
        amap[y, x] = 1
        crop = crop_around(p, amap)
        x0, y0, s = _engine_crop_box(400, 300, amap)
        assert crop.shape[:2] == (s, s) == (160, 160)
        assert 0 <= x0 <= 400 - s and 0 <= y0 <= 300 - s
        assert np.array_equal(crop, cv2.imread(str(p))[y0:y0 + s, x0:x0 + s])
