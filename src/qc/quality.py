"""Defect measurement (pixels -> millimetres) and the Pass / Rework / Scrap rules."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qc import config as C

SEVERITY = {d: i for i, d in enumerate(C.DECISIONS)}   # PASS < REWORK < SCRAP


@dataclass
class Measurement:
    area_mm2: float
    length_mm: float
    width_mm: float


def measure_mask(mask: np.ndarray, px_per_mm: float = C.PX_PER_MM, scale: float = 1.0) -> Measurement:
    """Measure one defect mask.

    mask:   boolean array (any resolution)
    scale:  camera pixels per mask pixel (e.g. crop_side / CROP_SIZE when the
            mask comes from a resized crop)
    length: long side of the minimum-area rotated rectangle around the defect
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return Measurement(0.0, 0.0, 0.0)
    area_px = len(xs) * scale ** 2
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    # +1 because a 1-pixel-wide line has a rotated-rect width of 0
    long_px, short_px = (max(w, h) + 1) * scale, (min(w, h) + 1) * scale
    return Measurement(area_px / px_per_mm ** 2, long_px / px_per_mm, short_px / px_per_mm)


def judge_defect(cls: str, m: Measurement, rules: dict) -> tuple[str, str]:
    """Return (verdict, reason) for one defect."""
    rule = rules.get(cls)
    if rule is None:
        return "SCRAP", f"unknown defect type '{cls}'"
    value = {"area_mm2": m.area_mm2, "length_mm": m.length_mm, "count": 1.0}[rule["metric"]]
    for band in rule["bands"]:
        if value < float(band["max"]):
            return band["verdict"], band["reason"]
    return rule["bands"][-1]["verdict"], rule["bands"][-1]["reason"]


def combine(verdicts) -> str:
    """Device decision = most severe defect verdict (no defects -> PASS)."""
    verdicts = list(verdicts)
    return max(verdicts, key=SEVERITY.get) if verdicts else "PASS"
