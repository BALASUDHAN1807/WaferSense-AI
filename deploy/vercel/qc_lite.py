"""Lightweight inference for the Vercel deployment.

Runs the SAME trained models as the full project (YOLO11n detector + YOLO11n-seg
defect segmenter, exported to ONNX) with only onnxruntime + numpy + Pillow, so it
fits in a serverless function. Same steps as src/qc/engine.py:

    detect devices -> square crop (8 % pad, 256 px) -> segment defects
    -> measure in mm (12 px/mm) -> quality rules -> PASS / REWORK / SCRAP
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
DEFECTS = CFG["defect_classes"]
SEVERITY = {"PASS": 0, "REWORK": 1, "SCRAP": 2}
DEFECT_RGB = {"crack": (230, 40, 40), "scratch": (230, 60, 230), "contamination": (255, 215, 0),
              "chip": (30, 140, 255), "missing_pin": (60, 220, 60)}
DECISION_RGB = {"PASS": (60, 190, 60), "REWORK": (255, 165, 0), "SCRAP": (230, 50, 50)}


# ------------------------------------------------------------- helpers

def letterbox(img: Image.Image, size: int):
    """Resize keeping aspect ratio and pad to size x size with grey 114 (as Ultralytics)."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, px, py


def nms(boxes, scores, iou=0.7):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ov = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou]
    return np.array(keep, int)


def xywh2xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def min_area_rect_sides(xs, ys):
    """Long and short side of the minimum-area rotated rectangle (like cv2.minAreaRect)."""
    pts = np.unique(np.stack([xs, ys], 1).astype(np.float64), axis=0)
    if len(pts) < 3:
        span = pts.max(0) - pts.min(0)
        return float(span.max()), float(span.min())
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        span = pts.max(0) - pts.min(0)
        return float(span.max()), float(span.min())
    edges = np.roll(hull, -1, 0) - hull
    ang = np.arctan2(edges[:, 1], edges[:, 0])
    best = None
    for a in np.unique(np.mod(ang, np.pi / 2)):
        c, s = np.cos(a), np.sin(a)
        rx, ry = hull[:, 0] * c + hull[:, 1] * s, -hull[:, 0] * s + hull[:, 1] * c
        w, h = rx.max() - rx.min(), ry.max() - ry.min()
        if best is None or w * h < best[0]:
            best = (w * h, w, h)
    return float(max(best[1:])), float(min(best[1:]))


def judge(cls, area_mm2, length_mm, rules):
    rule = rules.get(cls)
    if rule is None:
        return "SCRAP", f"unknown defect type '{cls}'"
    value = {"area_mm2": area_mm2, "length_mm": length_mm, "count": 1.0}[rule["metric"]]
    for band in rule["bands"]:
        mx = float("inf") if band["max"] is None else float(band["max"])
        if value < mx:
            return band["verdict"], band["reason"]
    return rule["bands"][-1]["verdict"], rule["bands"][-1]["reason"]


def combine(verdicts):
    v = list(verdicts)
    return max(v, key=SEVERITY.get) if v else "PASS"


# -------------------------------------------------------------- engine

class LiteEngine:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.det = ort.InferenceSession(str(HERE / "models" / "detector_yolo11n.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        self.seg = ort.InferenceSession(str(HERE / "models" / "segmenter_yolo11n_seg.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        self.rules = CFG["rules"]

    def detect(self, img: Image.Image):
        x, r, px, py = letterbox(img, CFG["det_imgsz"])
        out = self.det.run(None, {"images": x})[0][0].T          # (N, 5)
        keep = out[:, 4] >= CFG["det_conf"]
        out = out[keep]
        if not len(out):
            return []
        boxes = xywh2xyxy(out[:, :4])
        k = nms(boxes, out[:, 4])
        boxes, scores = boxes[k], out[k, 4]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        W, H = img.size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return list(zip(boxes.tolist(), scores.tolist()))

    @staticmethod
    def square_crop(arr: np.ndarray, box, size, pad):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(8, int(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad))))
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        padded = np.pad(arr, ((side, side), (side, side), (0, 0)), mode="edge")
        crop = padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
        im = Image.fromarray(crop).resize((size, size), Image.BOX if side > size else Image.BILINEAR)
        return im, (x0, y0, side)

    def segment(self, crop: Image.Image):
        size = CFG["seg_imgsz"]
        x = np.asarray(crop, np.float32).transpose(2, 0, 1)[None] / 255.0
        pred, protos = self.seg.run(None, {"images": x})
        pred, protos = pred[0].T, protos[0]                       # (N, 41), (32, 64, 64)
        nc = len(DEFECTS)
        cls_scores = pred[:, 4:4 + nc]
        cls, conf = cls_scores.argmax(1), cls_scores.max(1)
        keep = conf >= CFG["seg_conf"]
        if not keep.any():
            return []
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        boxes = xywh2xyxy(pred[:, :4])
        # per-class NMS (class offset trick, as Ultralytics)
        k = nms(boxes + cls[:, None] * 4096.0, conf)
        boxes, cls, conf, coef = boxes[k], cls[k], conf[k], pred[k, 4 + nc:]
        ph, pw = protos.shape[1:]
        m = 1 / (1 + np.exp(-(coef @ protos.reshape(32, -1)))).reshape(-1, ph, pw)
        out = []
        yy, xx = np.mgrid[0:size, 0:size]
        for i in range(len(k)):
            up = np.asarray(Image.fromarray(m[i].astype(np.float32)).resize((size, size), Image.BILINEAR))
            b = boxes[i]
            inside = (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
            mask = (up > 0.5) & inside
            if mask.sum() >= CFG["min_pixels"]:
                out.append((DEFECTS[int(cls[i])], float(conf[i]), mask))
        return out

    def inspect(self, img: Image.Image) -> tuple[dict, np.ndarray]:
        img = img.convert("RGB")
        arr = np.asarray(img)
        H, W = arr.shape[:2]
        t0 = time.perf_counter()
        dets = self.detect(img)
        t1 = time.perf_counter()
        devices, overlay_masks = [], []
        size, ppm = CFG["seg_imgsz"], CFG["px_per_mm"]
        for box, dconf in dets:
            crop, (x0, y0, side) = self.square_crop(arr, box, size, CFG["crop_pad"])
            scale = side / size
            defects = []
            for cls, p, mask in self.segment(crop):
                ys, xs = np.nonzero(mask)
                cx, cy = x0 + (xs + 0.5) * scale, y0 + (ys + 0.5) * scale
                inside = ((cx >= box[0] - 3) & (cx <= box[2] + 3) & (cy >= box[1] - 3) & (cy <= box[3] + 3)).mean()
                if inside < 0.5:                                  # belongs to a neighbouring device
                    continue
                long_px, short_px = min_area_rect_sides(xs, ys)
                area = len(xs) * scale ** 2 / ppm ** 2
                length, width = (long_px + 1) * scale / ppm, (short_px + 1) * scale / ppm
                verdict, reason = judge(cls, area, length, self.rules)
                defects.append({"cls": cls, "confidence": round(p, 4), "area_mm2": round(area, 3),
                                "length_mm": round(length, 3), "width_mm": round(width, 3),
                                "verdict": verdict, "reason": reason})
                overlay_masks.append((cls, x0, y0, side, mask))
            devices.append({"box": [round(v, 1) for v in box], "confidence": round(dconf, 4),
                            "decision": combine(d["verdict"] for d in defects), "defects": defects})
        t2 = time.perf_counter()
        counts = {d: 0 for d in SEVERITY}
        for d in devices:
            counts[d["decision"]] += 1
        result = {"overall": combine(d["decision"] for d in devices) if devices else "NO_DEVICE",
                  "counts": counts, "segmenter": "yolo (onnx)", "devices": devices,
                  "timings_ms": {"detection": round((t1 - t0) * 1000, 1),
                                 "segmentation": round((t2 - t1) * 1000, 1), "total": round((t2 - t0) * 1000, 1)}}
        return result, self.annotate(arr, result, overlay_masks)

    @staticmethod
    def annotate(arr, result, masks) -> np.ndarray:
        H, W = arr.shape[:2]
        out = arr.astype(np.float32).copy()
        for cls, x0, y0, side, mask in masks:
            big = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 127
            ys, xs = np.nonzero(big)
            ys, xs = ys + y0, xs + x0
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            out[ys[ok], xs[ok]] = out[ys[ok], xs[ok]] * 0.4 + np.array(DEFECT_RGB[cls]) * 0.6
        im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        for dev in result["devices"]:
            x1, y1, x2, y2 = dev["box"]
            col = DECISION_RGB[dev["decision"]]
            d.rectangle([x1, y1, x2, y2], outline=col, width=2)
            worst = [f for f in dev["defects"] if f["verdict"] == dev["decision"]]
            label = dev["decision"] + (f": {worst[0]['cls']}" if worst and dev["decision"] != "PASS" else "")
            tw = d.textlength(label, font=font)
            ty = y1 - 14 if y1 > 16 else y2 + 2
            d.rectangle([x1, ty, x1 + tw + 6, ty + 13], fill=col)
            d.text((x1 + 3, ty + 1), label, fill=(255, 255, 255), font=font)
        return np.asarray(im)


def to_png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
