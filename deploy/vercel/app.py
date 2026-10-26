"""Vercel deployment: FastAPI serving the trained ONNX models + the web dashboard.

    GET  /                  web dashboard (public/index.html, served by Vercel's CDN)
    GET  /api/health        model status
    POST /api/inspect       upload an image -> devices, defects, measurements, decisions,
                            annotated PNG (base64)
    GET  /api/selftest      inspect a built-in test frame and compare with the expected result
    GET  /api/samples       list of demo camera frames
    GET  /api/rules         quality rules in use
    POST /api/inspect-real  REAL photo (MVTec AD transistor): PatchCore heat-map + defect type
                            -> PASS / REWORK / SCRAP
    GET  /api/real-samples  demo photos (held out from every model trained on them)

Inspection history is kept in the visitor's browser: serverless functions have no
persistent disk, so the SQLite database is only used by the full local version.
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from qc_lite import CFG, LiteEngine, to_png_b64
from real_lite import RCFG, RealEngine

HERE = Path(__file__).resolve().parent
MAX_BYTES = 4 * 1024 * 1024          # Vercel request body limit is 4.5 MB

app = FastAPI(title="Semiconductor Device QC", version="1.0.0",
              description="YOLO11 detection + YOLO11-seg defect segmentation + PASS/REWORK/SCRAP rules")
_engine: LiteEngine | None = None
_real: RealEngine | None = None


def engine() -> LiteEngine:
    global _engine
    if _engine is None:                # loaded once per warm function instance
        _engine = LiteEngine()
    return _engine


def real_engine() -> RealEngine:
    global _real
    if _real is None:
        try:
            _real = RealEngine()
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(503, f"Real-image models not deployed yet ({e})")
    return _real


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 4 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Unsupported or corrupted image (use PNG/JPG/BMP)")
    if min(img.size) < 32 or max(img.size) > 4096:
        raise HTTPException(422, "Image must be between 32 and 4096 pixels per side")
    return img


@app.get("/api/health")
def health():
    try:
        engine()
        try:
            real_engine()
        except HTTPException:
            pass
        return {"status": "ok", "models": ["detector_yolo11n.onnx", "segmenter_yolo11n_seg.onnx",
                                           "patchcore_backbone.onnx", "transistor_seg.onnx"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)):
    img = await _read_image(file)
    result, annotated = engine().inspect(img)
    result["annotated_image_png_base64"] = to_png_b64(annotated)
    result["filename"] = file.filename
    return result


@app.get("/api/selftest")
def selftest():
    """Inspect a built-in test frame (3 devices: 2 PASS, 1 REWORK for contamination)."""
    result, _ = engine().inspect(Image.open(HERE / "selftest.png"))
    return {"expected": {"PASS": 2, "REWORK": 1, "SCRAP": 0}, "got": result["counts"],
            "ok": result["counts"] == {"PASS": 2, "REWORK": 1, "SCRAP": 0},
            "timings_ms": result["timings_ms"], "devices": result["devices"]}


@app.get("/api/samples")
def samples():
    return CFG["samples"]


@app.get("/api/rules")
def rules():
    return CFG["rules"]



@app.post("/api/inspect-real")
async def inspect_real(file: UploadFile = File(...)):
    """Real transistor photo (MVTec AD style, top view): anomaly heat-map, defect type, decision."""
    img = await _read_image(file)
    result, heat = real_engine().inspect(img)
    result["heatmap_png_base64"] = to_png_b64(heat)
    result["filename"] = file.filename
    return result


@app.get("/api/real-samples")
def real_samples():
    return RCFG["samples"]
