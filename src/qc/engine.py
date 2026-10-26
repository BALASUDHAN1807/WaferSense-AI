"""The inspection engine: camera frame -> devices -> defects -> measurements -> decision.

    1. YOLO11 detector finds every IC device in the frame.
    2. Each device is cut out as a square crop (256 x 256).
    3. A segmentation model (YOLO11-seg or U-Net) returns a pixel mask per defect.
    4. Each mask is measured in millimetres (area, length) using the camera calibration.
    5. configs/quality_rules.yaml turns measurements into PASS / REWORK / SCRAP.

Usage (CLI):
    python run_project.py inspect path/to/image.png [--segmenter yolo|unet]
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from qc import config as C
from qc.geometry import square_crop
from qc.quality import combine, judge_defect, measure_mask

DEFECT_COLOURS = {  # BGR
    "crack": (40, 40, 230), "scratch": (230, 60, 230), "contamination": (0, 215, 255),
    "chip": (255, 140, 30), "missing_pin": (60, 220, 60),
}
DECISION_COLOURS = {"PASS": (60, 190, 60), "REWORK": (0, 165, 255), "SCRAP": (50, 50, 230)}


@dataclass
class DefectResult:
    cls: str
    confidence: float
    area_mm2: float
    length_mm: float
    width_mm: float
    verdict: str
    reason: str
    polygon: list = field(default_factory=list)       # camera pixels [[x, y], ...]
    pixels: int = 0                                   # mask size in crop pixels


@dataclass
class DeviceResult:
    box: list
    confidence: float
    decision: str
    defects: list


@dataclass
class InspectionResult:
    devices: list
    segmenter: str
    timings_ms: dict

    @property
    def counts(self) -> dict:
        out = {d: 0 for d in C.DECISIONS}
        for dev in self.devices:
            out[dev.decision] += 1
        return out

    @property
    def overall(self) -> str:
        """Worst decision in the frame (NO_DEVICE when nothing was detected)."""
        return combine(d.decision for d in self.devices) if self.devices else "NO_DEVICE"

    def to_dict(self) -> dict:
        return {"overall": self.overall, "counts": self.counts, "segmenter": self.segmenter,
                "timings_ms": self.timings_ms, "devices": [asdict(d) for d in self.devices]}


def tuned_thresholds(segmenter: str) -> dict:
    """Confidence / minimum-size thresholds tuned on the VALIDATION split (see evaluate.py)."""
    if C.THRESHOLDS_FILE.exists():
        t = json.loads(C.THRESHOLDS_FILE.read_text()).get(segmenter)
        if t:
            return {"seg_conf": t["seg_conf"], "min_pixels": t["min_pixels"]}
    return {"seg_conf": C.SEG_CONF, "min_pixels": C.MIN_DEFECT_PIXELS}


def best_segmenter() -> str:
    if C.BEST_PIPELINE_FILE.exists():
        return json.loads(C.BEST_PIPELINE_FILE.read_text())["segmenter"]
    return "yolo" if C.YOLO_SEG_WEIGHTS.exists() else "unet"


class InspectionEngine:
    """Loads the models once; call .inspect(image) for every frame."""

    def __init__(self, segmenter: str | None = None, det_conf: float = C.DET_CONF,
                 seg_conf: float | None = None, min_pixels: int | None = None,
                 rules: dict | None = None, px_per_mm: float = C.PX_PER_MM):
        from ultralytics import YOLO

        if not C.DETECTOR_WEIGHTS.exists():
            raise FileNotFoundError(f"{C.DETECTOR_WEIGHTS.name} not found - run: python run_project.py train-detector")
        self.segmenter = segmenter or best_segmenter()
        if self.segmenter not in ("yolo", "unet"):
            raise ValueError("segmenter must be 'yolo' or 'unet'")
        tuned = tuned_thresholds(self.segmenter)
        self.seg_conf = tuned["seg_conf"] if seg_conf is None else seg_conf
        self.min_pixels = tuned["min_pixels"] if min_pixels is None else min_pixels
        self.det_conf, self.ppm = det_conf, px_per_mm
        self.rules = rules or C.load_rules()
        self.detector = YOLO(str(C.DETECTOR_WEIGHTS))
        if self.segmenter == "yolo":
            if not C.YOLO_SEG_WEIGHTS.exists():
                raise FileNotFoundError(f"{C.YOLO_SEG_WEIGHTS.name} not found - run: python run_project.py train-yoloseg")
            self.seg_model = YOLO(str(C.YOLO_SEG_WEIGHTS))
        else:
            if not C.UNET_WEIGHTS.exists():
                raise FileNotFoundError(f"{C.UNET_WEIGHTS.name} not found - run: python run_project.py train-unet")
            import torch
            from qc.unet import load_unet
            torch.set_grad_enabled(False)
            self.seg_model = load_unet()

    # --------------------------------------------------------- stages
    def detect(self, image):
        r = self.detector.predict(image, imgsz=C.DETECTOR["imgsz"], conf=self.det_conf, device=C.DEVICE,
                                  verbose=False)[0]
        return [(b.tolist(), float(c)) for b, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy())]

    def segment(self, crop) -> list[tuple[str, float, np.ndarray]]:
        """Defects in one 256x256 crop -> [(class, confidence, bool mask), ...]."""
        out = []
        if self.segmenter == "yolo":
            r = self.seg_model.predict(crop, imgsz=C.YOLO_SEG["imgsz"], conf=self.seg_conf, device=C.DEVICE,
                                       retina_masks=True, verbose=False)[0]
            if r.masks is not None:
                for m, c, p in zip(r.masks.data.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy()):
                    mask = m > 0.5
                    out.append((C.DEFECT_CLASSES[int(c)], float(p), mask))
        else:
            from qc.unet import to_tensor
            prob = self.seg_model(to_tensor(crop)).softmax(1)[0].numpy()
            pred = prob.argmax(0)
            for k, cls in enumerate(C.DEFECT_CLASSES, start=1):
                n, labels, stats, _ = cv2.connectedComponentsWithStats((pred == k).astype(np.uint8), 8)
                for j in range(1, n):
                    mask = labels == j
                    out.append((cls, float(prob[k][mask].mean()), mask))
        # identical post-processing for both models: confidence + minimum size (crop pixels)
        return [(c, p, m) for c, p, m in out if p >= self.seg_conf and m.sum() >= self.min_pixels]

    def inspect(self, image: np.ndarray) -> InspectionResult:
        if image is None or image.ndim != 3:
            raise ValueError("expected a colour image (H x W x 3)")
        t0 = time.perf_counter()
        boxes = self.detect(image)
        t1 = time.perf_counter()
        devices = []
        for box, conf in boxes:
            crop, info = square_crop(image, box)
            x1, y1, x2, y2 = box
            defects = []
            for cls, p, mask in self.segment(crop):
                cam = info.to_camera(mask, image.shape)
                inside = cam[int(max(0, y1 - 3)):int(y2 + 3), int(max(0, x1 - 3)):int(x2 + 3)].sum()
                if cam.sum() == 0 or inside / cam.sum() < 0.5:     # belongs to a neighbouring device
                    continue
                m = measure_mask(mask, self.ppm, scale=info.scale)
                verdict, reason = judge_defect(cls, m, self.rules)
                cnts, _ = cv2.findContours(cam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                poly = max(cnts, key=cv2.contourArea).reshape(-1, 2).tolist() if cnts else []
                defects.append(DefectResult(cls, round(p, 4), round(m.area_mm2, 3), round(m.length_mm, 3),
                                            round(m.width_mm, 3), verdict, reason, poly, int(mask.sum())))
            devices.append(DeviceResult([round(v, 1) for v in box], round(conf, 4),
                                        combine(d.verdict for d in defects), defects))
        t2 = time.perf_counter()
        return InspectionResult(devices, self.segmenter, {"detection": round((t1 - t0) * 1000, 1),
                                                          "segmentation": round((t2 - t1) * 1000, 1),
                                                          "total": round((t2 - t0) * 1000, 1)})


# ------------------------------------------------------------- drawing

def annotate(image: np.ndarray, result: InspectionResult) -> np.ndarray:
    """Draw defect masks, device boxes coloured by decision, and a summary banner."""
    out = image.copy()
    overlay = image.copy()
    for dev in result.devices:
        for d in dev.defects:
            if d.polygon:
                pts = np.array(d.polygon, np.int32)
                cv2.fillPoly(overlay, [pts], DEFECT_COLOURS[d.cls])
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for dev in result.devices:
        x1, y1, x2, y2 = [int(round(v)) for v in dev.box]
        col = DECISION_COLOURS[dev.decision]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        worst = [d for d in dev.defects if d.verdict == dev.decision]
        label = dev.decision + (f": {worst[0].cls}" if worst and dev.decision != "PASS" else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), col, -1)
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for d in dev.defects:
            if d.polygon:
                cv2.polylines(out, [np.array(d.polygon, np.int32)], True, DEFECT_COLOURS[d.cls], 1)
    c = result.counts
    banner = f"PASS {c['PASS']}  REWORK {c['REWORK']}  SCRAP {c['SCRAP']}  |  {result.timings_ms['total']:.0f} ms"
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (25, 25, 25), -1)
    cv2.putText(out, banner, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Inspect one camera image")
    p.add_argument("image")
    p.add_argument("--segmenter", choices=["yolo", "unet"], default=None)
    p.add_argument("--save", default=None, help="where to save the annotated image")
    p.add_argument("--no-db", action="store_true", help="do not log to the SQLite database")
    a = p.parse_args(argv)
    img = cv2.imread(a.image)
    if img is None:
        print(f"Error: cannot read image {a.image}")
        return 1
    engine = InspectionEngine(a.segmenter)
    res = engine.inspect(img)
    out = a.save or str(C.PRED_DIR / f"inspected_{__import__('pathlib').Path(a.image).stem}.png")
    C.ensure_dirs()
    cv2.imwrite(out, annotate(img, res))
    print(f"\nInspection ({res.segmenter} segmenter, {res.timings_ms['total']:.0f} ms)")
    print("-" * 60)
    for i, dev in enumerate(res.devices, 1):
        print(f"Device {i}: {dev.decision:6s}  (detector conf {dev.confidence:.2f})")
        for d in dev.defects:
            print(f"   - {d.cls:13s} area {d.area_mm2:6.2f} mm2  length {d.length_mm:5.2f} mm  "
                  f"-> {d.verdict:6s} ({d.reason})")
    print(f"\nFrame summary: {res.counts}  ->  overall {res.overall}")
    print(f"Annotated image: {out}")
    if not a.no_db:
        from qc import db
        iid = db.save_inspection(res, source=a.image, annotated_path=out)
        print(f"Logged to database as inspection #{iid}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
