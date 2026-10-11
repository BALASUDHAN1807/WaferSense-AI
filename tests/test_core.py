"""Tests for data generation, measurement, rules, crop geometry, U-Net and the database."""
import json

import numpy as np
import pytest

from qc import config as C
from qc import db, synth
from qc.geometry import crop_mask, square_crop
from qc.quality import Measurement, combine, judge_defect, measure_mask

RULES = C.load_rules()


# ------------------------------------------------------------- generator

@pytest.fixture(scope="module")
def scenes():
    rng = np.random.default_rng(123)
    return [synth.render_scene(rng) for _ in range(8)]


def test_scene_image_and_devices(scenes):
    for s in scenes:
        assert s.image.shape == (C.SCENE_SIZE, C.SCENE_SIZE, 3) and s.image.dtype == np.uint8
        assert 1 <= len(s.devices) <= 3
        for d in s.devices:
            x1, y1, x2, y2 = d.box
            assert 0 <= x1 < x2 <= C.SCENE_SIZE and 0 <= y1 < y2 <= C.SCENE_SIZE


def test_defect_masks_lie_inside_their_device(scenes):
    for s in scenes:
        for d in s.devices:
            for df in d.defects:
                assert df.cls in C.DEFECT_CLASSES
                assert df.mask.shape == (C.SCENE_SIZE, C.SCENE_SIZE)
                assert df.mask.sum() >= 3
                assert (df.mask & ~d.footprint).sum() == 0


def test_devices_do_not_overlap(scenes):
    for s in scenes:
        for i, a in enumerate(s.devices):
            for b in s.devices[i + 1:]:
                assert not (a.footprint & b.footprint).any()


def test_generator_is_deterministic():
    a = synth.render_scene(np.random.default_rng(7))
    b = synth.render_scene(np.random.default_rng(7))
    assert np.array_equal(a.image, b.image)


def test_ground_truth_decision_matches_rules(scenes):
    for s in scenes:
        for d in s.devices:
            gt = synth.device_ground_truth(d, RULES)
            assert gt["decision"] == combine(x["verdict"] for x in gt["defects"])


# --------------------------------------------------------- measurement

def test_measure_rectangle_in_mm():
    m = np.zeros((100, 100), bool)
    m[10:22, 10:46] = True              # 12 x 36 px = 1 x 3 mm at 12 px/mm
    r = measure_mask(m, px_per_mm=12)
    assert r.area_mm2 == pytest.approx(3.0, rel=1e-6)
    assert r.length_mm == pytest.approx(3.0, abs=0.1)
    assert r.width_mm == pytest.approx(1.0, abs=0.1)


def test_measure_scale_factor():
    m = np.zeros((50, 50), bool)
    m[0:6, 0:6] = True
    assert measure_mask(m, 12, scale=2.0).area_mm2 == pytest.approx(measure_mask(m, 12).area_mm2 * 4)


def test_measure_empty_mask():
    assert measure_mask(np.zeros((5, 5), bool)).area_mm2 == 0.0


# ---------------------------------------------------------------- rules

@pytest.mark.parametrize("cls,meas,expected", [
    ("crack", Measurement(0.1, 0.5, 0.1), "SCRAP"),
    ("missing_pin", Measurement(0.3, 1.2, 0.3), "SCRAP"),
    ("scratch", Measurement(0.05, 1.0, 0.1), "PASS"),
    ("scratch", Measurement(0.2, 3.0, 0.1), "REWORK"),
    ("scratch", Measurement(0.5, 6.0, 0.1), "SCRAP"),
    ("contamination", Measurement(0.3, 0.6, 0.5), "PASS"),
    ("contamination", Measurement(1.5, 1.5, 1.0), "REWORK"),
    ("contamination", Measurement(4.0, 3.0, 2.0), "SCRAP"),
    ("chip", Measurement(0.3, 0.7, 0.4), "PASS"),
    ("chip", Measurement(1.2, 1.5, 1.0), "SCRAP"),
])
def test_quality_rules(cls, meas, expected):
    assert judge_defect(cls, meas, RULES)[0] == expected


def test_combine_takes_worst():
    assert combine([]) == "PASS"
    assert combine(["PASS", "REWORK"]) == "REWORK"
    assert combine(["REWORK", "SCRAP", "PASS"]) == "SCRAP"


def test_unknown_defect_is_scrap():
    assert judge_defect("alien", Measurement(1, 1, 1), RULES)[0] == "SCRAP"


# -------------------------------------------------------------- geometry

def test_crop_roundtrip_keeps_mask_location():
    img = np.zeros((300, 300, 3), np.uint8)
    mask = np.zeros((300, 300), bool)
    mask[100:120, 140:150] = True
    crop, info = square_crop(img, (90, 80, 200, 190))
    assert crop.shape == (C.CROP_SIZE, C.CROP_SIZE, 3)
    back = info.to_camera(crop_mask(mask, info), img.shape)
    inter, union = (back & mask).sum(), (back | mask).sum()
    assert inter / union > 0.85


def test_crop_near_border_is_padded():
    img = np.full((100, 100, 3), 50, np.uint8)
    crop, info = square_crop(img, (0, 0, 40, 40))
    assert crop.shape == (C.CROP_SIZE, C.CROP_SIZE, 3) and info.x0 < 0


# ------------------------------------------------------------------ U-Net

def test_unet_output_shape():
    import torch
    from qc.unet import N_SEG_CLASSES, UNet, to_tensor
    m = UNet(base=8).eval()
    x = to_tensor(np.zeros((C.CROP_SIZE, C.CROP_SIZE, 3), np.uint8))
    with torch.no_grad():
        assert m(x).shape == (1, N_SEG_CLASSES, C.CROP_SIZE, C.CROP_SIZE)
        assert m(torch.zeros(2, 3, 128, 128)).shape == (2, N_SEG_CLASSES, 128, 128)  # patch size


# ---------------------------------------------------------------- database

def _fake_result():
    from qc.engine import DefectResult, DeviceResult, InspectionResult
    devs = [DeviceResult([1, 2, 30, 40], 0.9, "SCRAP",
                         [DefectResult("crack", 0.8, 1.2, 3.0, 0.2, "SCRAP", "crack in package body")]),
            DeviceResult([50, 50, 90, 90], 0.95, "PASS", [])]
    return InspectionResult(devs, "yolo", {"detection": 5, "segmentation": 10, "total": 15})


def test_database_roundtrip(tmp_path):
    path = tmp_path / "t.db"
    res = _fake_result()
    iid = db.save_inspection(res, source="unit-test", path=path)
    assert db.history(10, path)[0]["id"] == iid
    d = db.inspection_detail(iid, path)
    assert d["overall"] == "SCRAP" and len(d["devices"]) == 2
    assert d["devices"][0]["defects"][0]["cls"] == "crack"
    s = db.stats(path)
    assert s["devices"] == 2 and s["decisions"]["SCRAP"] == 1 and s["rates"]["PASS"] == 0.5
    assert db.inspection_detail(999, path) is None


def test_result_serialises_to_json():
    d = _fake_result().to_dict()
    json.dumps(d)
    assert d["overall"] == "SCRAP" and d["counts"] == {"PASS": 1, "REWORK": 0, "SCRAP": 1}
