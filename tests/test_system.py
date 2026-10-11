"""Tests that need the generated dataset and trained models (skipped otherwise)."""
import json

import cv2
import numpy as np
import pytest

from qc import config as C

needs_data = pytest.mark.skipif(not (C.SCENE_DIR / "data.yaml").exists(),
                                reason="dataset not generated - run: python run_project.py generate")
needs_models = pytest.mark.skipif(not (C.DETECTOR_WEIGHTS.exists() and
                                       (C.YOLO_SEG_WEIGHTS.exists() or C.UNET_WEIGHTS.exists())),
                                  reason="models not trained - run: python run_project.py train-all")


@needs_data
def test_dataset_validation_passes():
    from qc.eda import validate
    assert validate()["passed"]


@needs_data
def test_splits_do_not_share_frames():
    names = {s: {p.name for p in (C.SCENE_DIR / s / "images").glob("*.png")} for s in C.N_SCENES}
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"])


@needs_data
def test_annotations_have_decisions():
    ann = json.loads((C.ANNOTATION_DIR / "test.json").read_text())
    decisions = {d["decision"] for s in ann["scenes"] for d in s["devices"]}
    assert decisions == set(C.DECISIONS)


@pytest.fixture(scope="module")
def engine():
    from qc.engine import InspectionEngine
    return InspectionEngine()


@pytest.fixture(scope="module")
def sample():
    p = sorted(C.SAMPLE_DIR.glob("*.png"))[0]
    return p, cv2.imread(str(p))


@needs_models
def test_engine_inspects_sample(engine, sample):
    from qc.engine import annotate
    _, img = sample
    res = engine.inspect(img)
    assert len(res.devices) >= 1
    for dev in res.devices:
        assert dev.decision in C.DECISIONS
        for d in dev.defects:
            assert d.cls in C.DEFECT_CLASSES and d.area_mm2 > 0 and d.verdict in C.DECISIONS
    vis = annotate(img, res)
    assert vis.shape == img.shape
    json.dumps(res.to_dict())


@needs_models
def test_engine_finds_no_device_on_empty_conveyor(engine):
    blank = np.full((C.SCENE_SIZE, C.SCENE_SIZE, 3), (62, 78, 64), np.uint8)
    res = engine.inspect(blank)
    assert res.devices == [] and res.overall == "NO_DEVICE"


@needs_models
def test_engine_rejects_bad_input(engine):
    with pytest.raises(ValueError):
        engine.inspect(np.zeros((10, 10), np.uint8))


@needs_models
def test_api_endpoints(tmp_path, monkeypatch, sample):
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(C, "INSPECTION_IMG_DIR", tmp_path / "imgs")
    from fastapi.testclient import TestClient
    from api.main import app
    path, _ = sample
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/inspect", params={"include_image": True},
                        files={"file": (path.name, path.read_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["overall"] in C.DECISIONS and body["annotated_image_png_base64"]
        iid = body["inspection_id"]
        assert client.get(f"/inspections/{iid}").json()["id"] == iid
        assert client.get(f"/inspections/{iid}/image").status_code == 200
        assert client.get("/history").json()[0]["id"] == iid
        assert client.get("/stats").json()["inspections"] == 1
        assert client.post("/inspect", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 415
        assert client.post("/inspect", files={"file": ("x.png", b"", "image/png")}).status_code == 400
        assert client.get("/inspections/9999").status_code == 404
        assert "crack" in client.get("/rules").json()
