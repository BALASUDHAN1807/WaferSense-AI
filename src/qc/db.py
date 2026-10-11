"""SQLite storage of every inspection (results/inspections.db).

Tables
    inspections(id, created_at, source, segmenter, overall, n_devices, n_pass, n_rework, n_scrap,
                latency_ms, annotated_path)
    devices(id, inspection_id, idx, x1, y1, x2, y2, confidence, decision)
    defects(id, device_id, cls, confidence, area_mm2, length_mm, verdict, reason)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from qc import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT,
    segmenter TEXT,
    overall TEXT,
    n_devices INTEGER, n_pass INTEGER, n_rework INTEGER, n_scrap INTEGER,
    latency_ms REAL,
    annotated_path TEXT
);
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    idx INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL, confidence REAL, decision TEXT
);
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    cls TEXT, confidence REAL, area_mm2 REAL, length_mm REAL, verdict TEXT, reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_devices_insp ON devices(inspection_id);
CREATE INDEX IF NOT EXISTS idx_defects_dev ON defects(device_id);
"""


@contextmanager
def connect(path: Path | None = None):
    path = Path(path or C.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def save_inspection(result, source: str = "", annotated_path: str = "", path: Path | None = None) -> int:
    c = result.counts
    with connect(path) as con:
        cur = con.execute(
            "INSERT INTO inspections (created_at, source, segmenter, overall, n_devices, n_pass, n_rework,"
            " n_scrap, latency_ms, annotated_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), str(source), result.segmenter,
             result.overall, len(result.devices), c["PASS"], c["REWORK"], c["SCRAP"],
             result.timings_ms["total"], str(annotated_path)))
        iid = cur.lastrowid
        for k, dev in enumerate(result.devices):
            d = con.execute("INSERT INTO devices (inspection_id, idx, x1, y1, x2, y2, confidence, decision)"
                            " VALUES (?,?,?,?,?,?,?,?)", (iid, k, *dev.box, dev.confidence, dev.decision))
            for f in dev.defects:
                con.execute("INSERT INTO defects (device_id, cls, confidence, area_mm2, length_mm, verdict, reason)"
                            " VALUES (?,?,?,?,?,?,?)", (d.lastrowid, f.cls, f.confidence, f.area_mm2,
                                                        f.length_mm, f.verdict, f.reason))
    return iid


def history(limit: int = 50, path: Path | None = None) -> list[dict]:
    with connect(path) as con:
        rows = con.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def inspection_detail(iid: int, path: Path | None = None) -> dict | None:
    with connect(path) as con:
        row = con.execute("SELECT * FROM inspections WHERE id = ?", (iid,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["devices"] = []
        for dev in con.execute("SELECT * FROM devices WHERE inspection_id = ? ORDER BY idx", (iid,)):
            d = dict(dev)
            d["defects"] = [dict(x) for x in con.execute("SELECT * FROM defects WHERE device_id = ?", (dev["id"],))]
            out["devices"].append(d)
    return out


def stats(path: Path | None = None) -> dict:
    with connect(path) as con:
        t = con.execute("SELECT COUNT(*) n, COALESCE(SUM(n_devices),0) dev, COALESCE(SUM(n_pass),0) p,"
                        " COALESCE(SUM(n_rework),0) r, COALESCE(SUM(n_scrap),0) s, AVG(latency_ms) lat"
                        " FROM inspections").fetchone()
        defects = {r["cls"]: r["n"] for r in con.execute("SELECT cls, COUNT(*) n FROM defects GROUP BY cls")}
        reasons = [dict(r) for r in con.execute(
            "SELECT reason, COUNT(*) n FROM defects WHERE verdict != 'PASS' GROUP BY reason ORDER BY n DESC LIMIT 8")]
    n_dev = t["dev"] or 0
    return {
        "inspections": t["n"], "devices": n_dev,
        "decisions": {"PASS": t["p"], "REWORK": t["r"], "SCRAP": t["s"]},
        "rates": {k: (v / n_dev if n_dev else 0.0) for k, v in (("PASS", t["p"]), ("REWORK", t["r"]),
                                                                 ("SCRAP", t["s"]))},
        "defects_by_class": {c: defects.get(c, 0) for c in C.DEFECT_CLASSES},
        "top_reasons": reasons,
        "avg_latency_ms": round(t["lat"], 1) if t["lat"] is not None else None,
    }


def reset(path: Path | None = None) -> None:
    p = Path(path or C.DB_PATH)
    if p.exists():
        p.unlink()
