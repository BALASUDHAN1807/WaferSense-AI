"""Streamlit dashboard for the Semiconductor Device Quality Control system.

Talks to the FastAPI backend (default http://127.0.0.1:8000).  If the API is not
running it can fall back to running the models inside the dashboard ("local mode");
both modes write to the same SQLite database.

Run from the project root:
    python run_project.py dashboard        (http://localhost:8501)
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qc import config as C  # noqa: E402

st.set_page_config(page_title="Semiconductor QC", page_icon="🔬", layout="wide")

DECISION_STYLE = {"PASS": "#2e9e4f", "REWORK": "#e69500", "SCRAP": "#d63c3c", "NO_DEVICE": "#777777"}


# --------------------------------------------------------------- backends

class ApiBackend:
    def __init__(self, url):
        import httpx
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=60)

    def ok(self):
        try:
            return self.client.get(f"{self.url}/health").json().get("status") == "ok"
        except Exception:
            return False

    def inspect(self, png_bytes, name):
        r = self.client.post(f"{self.url}/inspect", params={"include_image": True},
                             files={"file": (name, png_bytes, "image/png")})
        r.raise_for_status()
        body = r.json()
        img = cv2.imdecode(np.frombuffer(base64.b64decode(body.pop("annotated_image_png_base64")), np.uint8),
                           cv2.IMREAD_COLOR)
        return body, img

    def stats(self):
        return self.client.get(f"{self.url}/stats").json()

    def history(self, n=200):
        return self.client.get(f"{self.url}/history", params={"limit": n}).json()


class LocalBackend:
    def __init__(self):
        from qc.engine import InspectionEngine
        self.engine = InspectionEngine()

    def ok(self):
        return True

    def inspect(self, png_bytes, name):
        from qc import db
        from qc.engine import annotate
        img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unsupported or corrupted image")
        res = self.engine.inspect(img)
        vis = annotate(img, res)
        C.INSPECTION_IMG_DIR.mkdir(parents=True, exist_ok=True)
        path = C.INSPECTION_IMG_DIR / f"local_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(path), vis)
        body = res.to_dict()
        body["inspection_id"] = db.save_inspection(res, source=name, annotated_path=str(path.relative_to(C.ROOT)))
        return body, vis

    def stats(self):
        from qc import db
        return db.stats()

    def history(self, n=200):
        from qc import db
        return db.history(n)


@st.cache_resource(show_spinner="Loading models...")
def local_backend():
    return LocalBackend()


def get_backend():
    url = st.session_state.get("api_url", os.environ.get("QC_API_URL", "http://127.0.0.1:8000"))
    api = ApiBackend(url)
    if api.ok():
        return api, f"FastAPI backend · {url}"
    try:
        return local_backend(), "Local mode (API not running - models loaded in the dashboard)"
    except Exception as e:
        st.error(f"Neither the API nor the local models are available: {e}\n\n"
                 "Train the models first: `python run_project.py all`")
        st.stop()


# ------------------------------------------------------------------- UI

def decision_badge(decision: str, big: bool = False) -> str:
    size = "1.6rem" if big else "0.95rem"
    return (f"<span style='background:{DECISION_STYLE[decision]};color:white;padding:2px 10px;"
            f"border-radius:6px;font-weight:600;font-size:{size}'>{decision}</span>")


def show_result(body: dict, original_bgr, annotated_bgr):
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), caption="Camera image", use_container_width=True)
    c2.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption="Inspection result",
             use_container_width=True)
    counts = body["counts"]
    st.markdown(f"**Frame result:** {decision_badge(body['overall'], True)} &nbsp; "
                f"{len(body['devices'])} device(s) · PASS {counts['PASS']} · REWORK {counts['REWORK']} · "
                f"SCRAP {counts['SCRAP']} · {body['timings_ms']['total']:.0f} ms · segmenter: "
                f"{body['segmenter']}", unsafe_allow_html=True)
    rows = []
    for i, dev in enumerate(body["devices"], 1):
        if not dev["defects"]:
            rows.append({"Device": i, "Decision": dev["decision"], "Defect": "-", "Area (mm²)": None,
                         "Length (mm)": None, "Verdict": "PASS", "Reason": "no defects found"})
        for d in dev["defects"]:
            rows.append({"Device": i, "Decision": dev["decision"], "Defect": d["cls"],
                         "Area (mm²)": d["area_mm2"], "Length (mm)": d["length_mm"],
                         "Confidence": d["confidence"], "Verdict": d["verdict"], "Reason": d["reason"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def sample_images():
    files = sorted(C.SAMPLE_DIR.glob("*.png"))
    test_dir = C.SCENE_DIR / "test" / "images"
    if test_dir.exists():
        files += sorted(test_dir.glob("*.png"))[12:]
    return files


def page_inspect(backend):
    st.subheader("Inspect a device image")
    left, right = st.columns([2, 1])
    up = left.file_uploader("Upload a camera image (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"])
    samples = sample_images()
    choice = right.selectbox("…or pick a sample frame", ["-"] + [p.name for p in samples[:40]])
    data, name = None, None
    if up is not None:
        data, name = up.getvalue(), up.name
    elif choice != "-":
        p = next(s for s in samples if s.name == choice)
        data, name = p.read_bytes(), p.name
    if data and st.button("Run inspection", type="primary"):
        original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if original is None:
            st.error("Unsupported or corrupted image.")
            return
        try:
            body, vis = backend.inspect(data, name)
        except Exception as e:
            st.error(f"Inspection failed: {e}")
            return
        show_result(body, original, vis)


def page_conveyor(backend):
    st.subheader("Conveyor simulation")
    st.caption("Feeds held-out test frames through the system one after another, like a camera "
               "above a production line.")
    c1, c2 = st.columns(2)
    n = c1.slider("Frames", 5, 60, 15)
    delay = c2.slider("Delay between frames (s)", 0.0, 2.0, 0.3)
    if not st.button("Start conveyor", type="primary"):
        return
    frames = sample_images()[:n]
    view, kpi, prog = st.empty(), st.empty(), st.progress(0.0)
    totals = {"PASS": 0, "REWORK": 0, "SCRAP": 0}
    for k, p in enumerate(frames, 1):
        body, vis = backend.inspect(p.read_bytes(), p.name)
        for d in totals:
            totals[d] += body["counts"][d]
        view.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{p.name} - {body['overall']}", width=560)
        n_dev = sum(totals.values()) or 1
        kpi.markdown(" &nbsp; ".join(f"{decision_badge(d)} {v} ({v / n_dev:.0%})" for d, v in totals.items())
                     + f" &nbsp; · {k}/{len(frames)} frames", unsafe_allow_html=True)
        prog.progress(k / len(frames))
        time.sleep(delay)
    st.success("Conveyor run finished - see the Analytics tab for the totals.")


def page_analytics(backend):
    st.subheader("Production analytics")
    s = backend.stats()
    if not s["inspections"]:
        st.info("No inspections yet - run some in the Inspect or Conveyor tab.")
        return
    k = st.columns(5)
    k[0].metric("Frames inspected", s["inspections"])
    k[1].metric("Devices", s["devices"])
    k[2].metric("Pass rate", f"{s['rates']['PASS']:.1%}")
    k[3].metric("Rework rate", f"{s['rates']['REWORK']:.1%}")
    k[4].metric("Scrap rate", f"{s['rates']['SCRAP']:.1%}")
    c1, c2 = st.columns(2)
    c1.markdown("**Decisions**")
    c1.bar_chart(pd.Series(s["decisions"], name="devices"), color="#3b6ea5")
    c2.markdown("**Defects found by type**")
    c2.bar_chart(pd.Series(s["defects_by_class"], name="defects"), color="#e07b39")
    if s["top_reasons"]:
        st.markdown("**Most common reasons for rework / scrap**")
        st.dataframe(pd.DataFrame(s["top_reasons"]), hide_index=True, use_container_width=True)
    st.markdown("**Recent inspections**")
    h = pd.DataFrame(backend.history(100))
    cols = ["id", "created_at", "source", "overall", "n_devices", "n_pass", "n_rework", "n_scrap",
            "latency_ms", "segmenter"]
    st.dataframe(h[[c for c in cols if c in h]], hide_index=True, use_container_width=True)
    if s["avg_latency_ms"]:
        st.caption(f"Average processing time: {s['avg_latency_ms']:.0f} ms per frame (CPU)")


def page_rules():
    st.subheader("Quality rules")
    st.caption(f"Loaded from `{C.RULES_FILE.relative_to(C.ROOT)}` - edit that file to change the limits.")
    rows = []
    for cls, r in C.load_rules().items():
        lo = 0.0
        for b in r["bands"]:
            hi = float(b["max"])
            rng = "any" if r["metric"] == "count" else (f"< {hi:g}" if lo == 0 else
                                                        (f"≥ {lo:g}" if hi == float("inf") else f"{lo:g} – {hi:g}"))
            rows.append({"Defect": cls, "Metric": r["metric"], "Range": rng, "Verdict": b["verdict"],
                         "Reason": b["reason"]})
            lo = hi
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main():
    st.title("🔬 Semiconductor Device Quality Control")
    st.caption("Computer-vision inspection of IC packages: YOLO11 device detection → defect segmentation → "
               "measurement in mm → PASS / REWORK / SCRAP. Academic prototype trained on synthetic images.")
    with st.sidebar:
        st.header("Settings")
        st.text_input("API URL", value=os.environ.get("QC_API_URL", "http://127.0.0.1:8000"), key="api_url")
        backend, label = get_backend()
        st.success(label)
        st.markdown("**Legend**")
        for d in C.DECISIONS:
            st.markdown(decision_badge(d), unsafe_allow_html=True)
    tabs = st.tabs(["Inspect", "Conveyor simulation", "Analytics", "Quality rules"])
    with tabs[0]:
        page_inspect(backend)
    with tabs[1]:
        page_conveyor(backend)
    with tabs[2]:
        page_analytics(backend)
    with tabs[3]:
        page_rules()


main()
