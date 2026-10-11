"""FastAPI backend for the Semiconductor Device Quality Control system.

Endpoints
    GET  /health                       model + database status
    POST /inspect                      upload an image -> devices, defects, measurements, decisions
                                       (?include_image=true adds the annotated image as base64 PNG)
    GET  /history?limit=50             latest inspections from SQLite
    GET  /inspections/{id}             one inspection with all devices and defects
    GET  /inspections/{id}/image       annotated image of that inspection
    GET  /stats                        totals, Pass/Rework/Scrap rates, defects per class
    GET  /rules                        the quality rules currently applied

Run from the project root:
    python run_project.py api            (http://127.0.0.1:8000, docs at /docs)
"""
from __future__ import annotations

import base64
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qc import config as C  # noqa: E402
from qc import db  # noqa: E402
from qc.engine import InspectionEngine, annotate  # noqa: E402

STATE: dict = {"engine": None, "error": None}
LOCK = threading.Lock()          # the models are not re-entrant; inspect one frame at a time
MAX_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["engine"] = InspectionEngine()     # loads detector + best segmenter once
    except Exception as e:                        # keep the API up and report the problem
        STATE["error"] = str(e)
    yield


app = FastAPI(title="Semiconductor Device QC API", version="1.0.0", lifespan=lifespan,
              description="YOLO11 detection + defect segmentation + Pass/Rework/Scrap rules")


def _engine() -> InspectionEngine:
    if STATE["engine"] is None:
        raise HTTPException(503, f"Models not loaded: {STATE['error']}")
    return STATE["engine"]


@app.get("/health")
def health():
    eng = STATE["engine"]
    return {"status": "ok" if eng else "degraded", "segmenter": eng.segmenter if eng else None,
            "error": STATE["error"],
            "database": str(C.DB_PATH.relative_to(C.ROOT) if C.ROOT in C.DB_PATH.parents else C.DB_PATH)}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), include_image: bool = Query(False),
                  save: bool = Query(True, description="log the inspection to the database")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "Unsupported or corrupted image (use PNG/JPG/BMP)")
    engine = _engine()
    with LOCK:
        result = engine.inspect(image)
    vis = annotate(image, result)
    body = result.to_dict()
    if save:
        C.INSPECTION_IMG_DIR.mkdir(parents=True, exist_ok=True)
        path = C.INSPECTION_IMG_DIR / f"{uuid4().hex[:12]}.png"
        cv2.imwrite(str(path), vis)
        stored = path.relative_to(C.ROOT) if C.ROOT in path.parents else path
        body["inspection_id"] = db.save_inspection(result, source=file.filename or "upload",
                                                   annotated_path=str(stored))
    if include_image:
        ok, png = cv2.imencode(".png", vis)
        body["annotated_image_png_base64"] = base64.b64encode(png.tobytes()).decode()
    return body


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=1000)):
    return db.history(limit)


@app.get("/inspections/{iid}")
def inspection(iid: int):
    d = db.inspection_detail(iid)
    if d is None:
        raise HTTPException(404, "Inspection not found")
    return d


@app.get("/inspections/{iid}/image")
def inspection_image(iid: int):
    d = db.inspection_detail(iid)
    if d is None or not d.get("annotated_path"):
        raise HTTPException(404, "Image not found")
    path = (C.ROOT / d["annotated_path"]).resolve()     # absolute paths stay absolute
    allowed = (C.ROOT.resolve(), C.INSPECTION_IMG_DIR.resolve())
    if not any(a in path.parents for a in allowed) or path.suffix != ".png" or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/stats")
def stats():
    return db.stats()


@app.get("/rules")
def rules():
    """Quality rules; an open upper limit (.inf in the YAML) is returned as null."""
    out = C.load_rules()
    for rule in out.values():
        for band in rule["bands"]:
            if band["max"] == float("inf"):
                band["max"] = None
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
