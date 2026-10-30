"""Real-image (MVTec AD transistor) inspection for the Vercel deployment.

Same two stages as src/qc/real/ (evaluated with 5-fold cross-validation in the report):

    PatchCore (learned from defect-free photos only) -> anomaly score + heat-map
        defective = PatchCore score >= t_pc  OR  YOLO11n-seg confidence >= t_seg  (score fusion)
        not defective -> PASS
        defective     -> defect type -> transistor rules -> REWORK / SCRAP
                         type = YOLO11s-cls on a crop around PatchCore's hottest point (report 12.6),
                                or YOLO11n-seg's top detection when the classifier is not exported

Only onnxruntime + numpy + Pillow are used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from qc_lite import nms, xywh2xyxy

HERE = Path(__file__).resolve().parent
_CFG_FILE = HERE / "real_config.json"
RCFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {"samples": []}
TYPE_RGB = {"bent_lead": (255, 165, 0), "cut_lead": (230, 50, 50), "damaged_case": (230, 60, 230),
            "misplaced": (30, 140, 255)}


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (same as the training code)."""
    k = int(4 * sigma + 1) | 1
    t = np.arange(k, dtype=np.float32) - k // 2
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    p = k // 2
    x = np.pad(m, ((0, 0), (p, p)), mode="reflect")
    x = np.stack([np.convolve(r, g, mode="valid") for r in x])
    x = np.pad(x, ((p, p), (0, 0)), mode="reflect")
    return np.stack([np.convolve(c, g, mode="valid") for c in x.T]).T


def _jet(v: np.ndarray) -> np.ndarray:
    """0..1 -> RGB (blue -> green -> yellow -> red)."""
    v = np.clip(v, 0, 1)[..., None]
    stops = np.array([[40, 70, 200], [40, 190, 120], [250, 220, 60], [220, 30, 30]], np.float32)
    pos = v * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = pos - i
    return (stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f).astype(np.uint8)


class RealEngine:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        m = HERE / "models"
        self.backbone = ort.InferenceSession(str(m / "patchcore_backbone.onnx"), opts,
                                             providers=["CPUExecutionProvider"])
        self.seg = ort.InferenceSession(str(m / "transistor_seg.onnx"), opts, providers=["CPUExecutionProvider"])
        self.bank = np.load(m / "patchcore_bank.npy").astype(np.float32)
        self.bank_sq = (self.bank ** 2).sum(1)
        self.pc, self.sc, self.rules = RCFG["patchcore"], RCFG["seg"], RCFG["rules"]
        self.tc = RCFG.get("type_cls")
        self.cls = (ort.InferenceSession(str(m / "transistor_type_cls.onnx"), opts, providers=["CPUExecutionProvider"])
                    if self.tc and (m / "transistor_type_cls.onnx").exists() else None)

    def anomaly(self, img: Image.Image):
        s = self.pc["img_size"]
        x = np.asarray(img.resize((s, s), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
        f = self.backbone.run(None, {"images": x})[0][0]                 # C, H, W
        C_, H, W = f.shape
        q = f.reshape(C_, -1).T
        d2 = (q ** 2).sum(1)[:, None] + self.bank_sq[None] - 2 * q @ self.bank.T
        d = np.sqrt(np.clip(d2.min(1), 0, None)).reshape(H, W)
        score = float(d.max())
        up = np.asarray(Image.fromarray(d.astype(np.float32)).resize((s, s), Image.BILINEAR))
        return score, _blur(up, self.pc["blur_sigma"])

    def defect_type(self, img: Image.Image):
        s = self.sc["img_size"]
        x = np.asarray(img.resize((s, s), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
        pred, protos = self.seg.run(None, {"images": x})
        pred, protos = pred[0].T, protos[0]
        nc = len(self.sc["classes"])
        cs = pred[:, 4:4 + nc]
        cls, conf = cs.argmax(1), cs.max(1)
        keep = conf >= self.sc["conf"]
        if not keep.any():
            return []
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        boxes = xywh2xyxy(pred[:, :4])
        k = nms(boxes + cls[:, None] * 4096.0, conf)
        ph, pw = protos.shape[1:]
        coef = pred[k, 4 + nc:]
        masks = 1 / (1 + np.exp(-(coef @ protos.reshape(32, -1)))).reshape(-1, ph, pw)
        out = []
        for j, i in enumerate(k):
            b = boxes[i] / s                                              # normalised box
            out.append({"cls": self.sc["classes"][int(cls[i])], "confidence": float(conf[i]),
                        "box": b.clip(0, 1).tolist(), "mask": masks[j]})
        return sorted(out, key=lambda d: -d["confidence"])

    def crop_box(self, img: Image.Image, amap: np.ndarray):
        """Square (normalised x1, y1, x2, y2) around the anomaly map's maximum, as in training."""
        W, H = img.size
        y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
        cy, cx = (y + 0.5) / amap.shape[0] * H, (x + 0.5) / amap.shape[1] * W
        s = int(self.tc["crop_fraction"] * max(H, W))
        x0 = int(np.clip(cx - s / 2, 0, W - s)); y0 = int(np.clip(cy - s / 2, 0, H - s))
        return x0, y0, s

    def classify(self, img: Image.Image, amap: np.ndarray):
        x0, y0, s = self.crop_box(img, amap)
        n = self.tc["img_size"]
        crop = img.crop((x0, y0, x0 + s, y0 + s)).resize((n, n), Image.BILINEAR)
        x = np.asarray(crop, np.float32).transpose(2, 0, 1)[None] / 255.0
        p = self.cls.run(None, {self.cls.get_inputs()[0].name: x})[0][0]
        W, H = img.size
        k = int(p.argmax())
        return {"cls": self.tc["classes"][k], "confidence": float(p[k]),
                "probs": {c: round(float(v), 4) for c, v in zip(self.tc["classes"], p)},
                "box": [x0 / W, y0 / H, (x0 + s) / W, (y0 + s) / H]}

    def inspect(self, img: Image.Image):
        t0 = time.perf_counter()
        img = img.convert("RGB")
        score, amap = self.anomaly(img)
        t1 = time.perf_counter()
        dets = self.defect_type(img)
        t2 = time.perf_counter()
        thr = self.pc["threshold"]
        gate = self.sc.get("gate_conf")                                  # score fusion (report 12.4)
        seg_hit = bool(dets) and gate is not None and dets[0]["confidence"] >= gate
        anomalous = score >= thr or seg_hit
        typ = self.classify(img, amap) if anomalous and self.cls is not None else None
        t3 = time.perf_counter()
        if not anomalous:
            dtype, decision, reason = None, "PASS", "no anomaly and no confident defect detection"
        else:
            dtype = typ["cls"] if typ else (dets[0]["cls"] if dets else "unknown")
            rule = self.rules.get(dtype, self.rules["unknown"])
            decision, reason = rule["verdict"], rule["reason"]
        result = {
            "decision": decision, "reason": reason, "defect_type": dtype,
            "anomaly_score": round(score, 3), "threshold": round(thr, 3), "anomalous": bool(anomalous),
            "type_source": None if not anomalous else ("YOLO11s-cls (PatchCore crop)" if typ else "YOLO11-seg"),
            "type_classifier": typ,
            "flagged_by": ([n for n, v in (("PatchCore", score >= thr), ("YOLO11-seg", seg_hit)) if v]),
            "detections": [{k: v for k, v in d.items() if k != "mask"} for d in dets[:3]],
            "timings_ms": {"patchcore": round((t1 - t0) * 1e3, 1), "type_model": round((t2 - t1) * 1e3, 1),
                           "type_classifier": round((t3 - t2) * 1e3, 1),
                           "total": round((time.perf_counter() - t0) * 1e3, 1)},
        }
        shown = ([dict(typ)] if typ else dets) if anomalous else []
        return result, self.render(img, amap, shown, result)

    def render(self, img, amap, dets, result, size=384):
        base = np.asarray(img.resize((size, size), Image.BILINEAR), np.float32)
        a = np.asarray(Image.fromarray(amap.astype(np.float32)).resize((size, size), Image.BILINEAR))
        thr = result["threshold"]
        norm = np.clip((a - 0.7 * thr) / (0.4 * thr), 0, 1)               # fixed scale: comparable across images
        heat = (0.55 * base + 0.45 * _jet(norm)).astype(np.uint8)
        im = Image.fromarray(heat)
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        if dets:
            t = dets[0]
            x1, y1, x2, y2 = [v * size for v in t["box"]]
            col = TYPE_RGB.get(t["cls"], (255, 255, 255))
            d.rectangle([x1, y1, x2, y2], outline=col, width=3)
            label = f'{t["cls"].replace("_", " ")} {t["confidence"]:.2f}'
            ty = y1 - 14 if y1 > 16 else y2 + 2
            d.rectangle([x1, ty, x1 + d.textlength(label, font=font) + 6, ty + 13], fill=col)
            d.text((x1 + 3, ty + 1), label, fill=(255, 255, 255), font=font)
        return np.asarray(im)
