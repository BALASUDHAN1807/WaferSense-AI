"""Synthetic camera images of packaged semiconductor devices (QFP-style ICs).

Each scene is a top-down camera frame of a conveyor/tray carrying 1-3 IC packages.
A package is a dark mould-compound body with marking text, a pin-1 dot and
leads (pins) on all four sides.  Five defect types can be drawn on a device,
and the generator returns the EXACT pixel mask of every defect, so detection
boxes, segmentation masks, measurements and the ground-truth Pass/Rework/Scrap
decision come for free.

    crack          dark jagged line (with branches) through the body
    scratch        thin bright straight/slightly curved line on the body
    contamination  residue / flux / particle blob
    chip           piece broken off the body edge or corner
    missing_pin    one or more leads absent

The images are SYNTHETIC - they imitate the look of real IC inspection images
but are not photographs of real devices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from qc import config as C
from qc.quality import combine, judge_defect, measure_mask


@dataclass
class Defect:
    cls: str
    mask: np.ndarray            # bool, scene resolution


@dataclass
class Device:
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 (scene pixels, inclusive-exclusive)
    footprint: np.ndarray            # bool mask of the whole device (body + pins)
    body_mm: float
    angle: float
    defects: list[Defect] = field(default_factory=list)


@dataclass
class Scene:
    image: np.ndarray                # H x W x 3 uint8, BGR
    devices: list[Device]


# --------------------------------------------------------------- utilities

def _texture(rng, shape, sigma, amp):
    n = rng.normal(0, 1, shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    n /= (n.std() + 1e-6)
    return n * amp


def _poly_line(rng, start, angle, length, step=3.0, jitter=0.35):
    pts = [np.array(start, dtype=np.float32)]
    a = angle
    for _ in range(max(2, int(length / step))):
        a += rng.normal(0, jitter)
        pts.append(pts[-1] + step * np.array([np.cos(a), np.sin(a)], dtype=np.float32))
    return np.array(pts)


def _mask_from_lines(shape, lines, thickness):
    m = np.zeros(shape, np.uint8)
    for pts in lines:
        cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, thickness, cv2.LINE_8)
    return m.astype(bool)


# -------------------------------------------------------------- device art

class _DeviceCanvas:
    """Renders one device in a local, axis-aligned canvas."""

    def __init__(self, rng, ppm):
        self.rng = rng
        self.body_mm = rng.uniform(7.0, 12.0)
        self.B = int(round(self.body_mm * ppm))
        self.pin_len = int(round(rng.uniform(1.0, 1.4) * ppm))
        self.pin_w = max(3, int(round(rng.uniform(0.28, 0.36) * ppm)))
        pitch = rng.uniform(0.70, 0.85) * ppm
        self.n_pins = max(5, int((self.B * 0.84) / pitch))
        self.pitch = pitch
        self.m = self.pin_len + 4
        self.S = self.B + 2 * self.m
        self.x0 = self.y0 = self.m
        self.x1 = self.y1 = self.m + self.B
        self.img = np.zeros((self.S, self.S, 3), np.float32)
        self.alpha = np.zeros((self.S, self.S), bool)
        self.body = np.zeros((self.S, self.S), bool)
        self.defects: list[tuple[str, np.ndarray]] = []

    # ---- base package
    def pin_rects(self):
        """(side, index) -> (x1, y1, x2, y2) for every lead."""
        rects = {}
        off = (self.B - (self.n_pins - 1) * self.pitch) / 2
        hw = self.pin_w / 2
        for i in range(self.n_pins):
            c = off + i * self.pitch
            rects[("top", i)] = (self.x0 + c - hw, self.y0 - self.pin_len, self.x0 + c + hw, self.y0)
            rects[("bottom", i)] = (self.x0 + c - hw, self.y1, self.x0 + c + hw, self.y1 + self.pin_len)
            rects[("left", i)] = (self.x0 - self.pin_len, self.y0 + c - hw, self.x0, self.y0 + c + hw)
            rects[("right", i)] = (self.x1, self.y0 + c - hw, self.x1 + self.pin_len, self.y0 + c + hw)
        return {k: tuple(int(round(v)) for v in r) for k, r in rects.items()}

    def draw_base(self, missing: set):
        rng = self.rng
        # body: dark mould compound with fine texture and soft rounded corners
        base = rng.uniform(30, 52)
        tint = np.array([base + rng.uniform(0, 7), base + rng.uniform(0, 4), base], np.float32)
        r = max(2, self.B // 40)
        body = np.zeros((self.S, self.S), np.uint8)
        cv2.rectangle(body, (self.x0 + r, self.y0), (self.x1 - 1 - r, self.y1 - 1), 1, -1)
        cv2.rectangle(body, (self.x0, self.y0 + r), (self.x1 - 1, self.y1 - 1 - r), 1, -1)
        for cx, cy in ((self.x0 + r, self.y0 + r), (self.x1 - 1 - r, self.y0 + r),
                       (self.x0 + r, self.y1 - 1 - r), (self.x1 - 1 - r, self.y1 - 1 - r)):
            cv2.circle(body, (cx, cy), r, 1, -1)
        self.body = body.astype(bool)
        tex = _texture(rng, self.img.shape, 1.2, 3.0) + _texture(rng, self.img.shape, 8, 4.0)
        for ch in range(3):
            self.img[..., ch][self.body] = (tint[ch] + tex)[self.body]
        self.alpha |= self.body

        # pin-1 dimple and marking text
        d = int(self.B * 0.06)
        cv2.circle(self.img, (self.x0 + 2 * d, self.y0 + 2 * d), max(2, d),
                   tuple(float(v) for v in tint - 12), -1, cv2.LINE_AA)
        text_col = float(rng.uniform(120, 165))
        lines = [f"WS{rng.integers(10, 99)}F{rng.integers(100, 999)}",
                 f"{rng.integers(2400, 2652)} {chr(65 + rng.integers(0, 26))}{rng.integers(1, 9)}"]
        scale = self.B / 260
        ink = np.zeros((self.S, self.S), np.uint8)
        for k, txt in enumerate(lines):
            y = int(self.y0 + self.B * (0.45 + 0.17 * k))
            cv2.putText(ink, txt, (int(self.x0 + self.B * 0.18), y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, 255, max(1, int(scale * 2)), cv2.LINE_AA)
        a = (ink.astype(np.float32) / 255.0)[..., None] * 0.85
        self.img = self.img * (1 - a) + text_col * a

        # leads: silver with a darker outline and a highlight
        silver = rng.uniform(175, 215)
        for key, (a, b, c, d2) in self.pin_rects().items():
            if key in missing:
                continue
            cv2.rectangle(self.img, (a, b), (c - 1, d2 - 1), (silver - 35,) * 3, -1)
            cv2.rectangle(self.img, (a + 1, b + 1), (c - 2, d2 - 2), (silver,) * 3, -1)
            self.alpha[b:d2, a:c] = True

    # ---- defects (all return the exact mask)
    def _random_body_point(self, margin=0.12):
        lo, hi = margin, 1 - margin
        return (self.x0 + self.B * self.rng.uniform(lo, hi), self.y0 + self.B * self.rng.uniform(lo, hi))

    def crack(self):
        rng = self.rng
        L = self.B * rng.uniform(0.3, 0.8)
        start = self._random_body_point(0.1)
        lines = [_poly_line(rng, start, rng.uniform(0, 2 * np.pi), L, jitter=0.45)]
        for _ in range(rng.integers(0, 3)):
            base = lines[0][rng.integers(len(lines[0]) // 4, len(lines[0]))]
            lines.append(_poly_line(rng, base, rng.uniform(0, 2 * np.pi), L * rng.uniform(0.2, 0.45),
                                    jitter=0.5))
        thick = int(rng.choice([1, 2]))
        core = _mask_from_lines(self.body.shape, lines, thick) & self.body
        halo = _mask_from_lines(self.body.shape, lines, thick + 2) & self.body & ~core
        self.img[halo] = self.img[halo] * 0.6 + rng.uniform(55, 75) * 0.4
        self.img[core] = rng.uniform(6, 16)
        return core | halo

    def scratch(self, length_mm, ppm):
        rng = self.rng
        L = min(length_mm * ppm, self.B * 0.9)
        ang = rng.uniform(0, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        # centre chosen so the whole line stays on the body
        half = np.abs(d) * L / 2
        cx = rng.uniform(self.x0 + half[0] + 3, max(self.x0 + half[0] + 4, self.x1 - half[0] - 3))
        cy = rng.uniform(self.y0 + half[1] + 3, max(self.y0 + half[1] + 4, self.y1 - half[1] - 3))
        t = np.linspace(-L / 2, L / 2, max(3, int(L)))
        bend = rng.uniform(-0.004, 0.004)
        pts = np.stack([cx + t * d[0] - bend * t ** 2 * d[1], cy + t * d[1] + bend * t ** 2 * d[0]], 1)
        m = _mask_from_lines(self.body.shape, [pts], int(rng.choice([1, 1, 2]))) & self.body
        bright = rng.uniform(95, 150)
        a = rng.uniform(0.65, 0.9)
        self.img[m] = self.img[m] * (1 - a) + bright * a
        return m

    def contamination(self, area_mm2, ppm):
        rng = self.rng
        target = area_mm2 * ppm ** 2
        cx, cy = self._random_body_point(0.15)
        m = np.zeros(self.body.shape, np.uint8)
        r = np.sqrt(target / np.pi)
        for _ in range(rng.integers(1, 4)):
            ox, oy = rng.normal(0, r * 0.45, 2)
            ax = (max(2, int(r * rng.uniform(0.55, 1.1))), max(2, int(r * rng.uniform(0.55, 1.1))))
            cv2.ellipse(m, (int(cx + ox), int(cy + oy)), ax, rng.uniform(0, 180), 0, 360, 1, -1)
        noise = _texture(rng, m.shape, max(1.0, r / 4), 1.0)
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, r / 5))
        m = ((soft + 0.18 * noise) > 0.5) & (soft > 0.2) & self.alpha
        kind = rng.choice(["residue", "flux", "particle"])
        colour = {"residue": (195, 200, 205), "flux": (60, 140, 175),
                  "particle": (95, 105, 120)}[kind]
        a = rng.uniform(0.7, 0.95)
        self.img[m] = self.img[m] * (1 - a) + np.array(colour, np.float32) * a
        return m

    def chip(self, area_mm2, ppm):
        rng = self.rng
        target = area_mm2 * ppm ** 2
        corner = rng.random() < 0.5
        if corner:
            cx, cy = [(self.x0, self.y0), (self.x1, self.y0), (self.x0, self.y1), (self.x1, self.y1)][
                rng.integers(0, 4)]
            r = np.sqrt(2 * target)
        else:
            side = rng.integers(0, 4)
            t = rng.uniform(0.15, 0.85) * self.B
            cx, cy = [(self.x0 + t, self.y0), (self.x0 + t, self.y1),
                      (self.x0, self.y0 + t), (self.x1, self.y0 + t)][side]
            r = np.sqrt(2 * target / np.pi)
        n = 9
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1)
        rad = r * rng.uniform(0.7, 1.3, n)
        poly = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
        m = np.zeros(self.body.shape, np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        m = m.astype(bool) & self.body
        if rng.random() < 0.5:   # material completely gone: background shows through
            self.alpha &= ~m
            self.body &= ~m
        else:                    # fractured, lighter compound exposed
            self.img[m] = rng.uniform(85, 120) + _texture(rng, m.shape, 0.8, 9)[m][:, None]
        return m

    def render(self, ppm, specs):
        """specs: list of (cls, size) - size is mm (scratch), mm^2 (contam./chip) or #pins."""
        missing, masks = set(), []
        pins = list(self.pin_rects().keys())
        for cls, size in specs:
            if cls == "missing_pin":
                side = pins[self.rng.integers(0, len(pins))][0]
                start = self.rng.integers(0, self.n_pins - int(size) + 1)
                keys = {(side, start + k) for k in range(int(size))}
                missing |= keys
                m = np.zeros(self.body.shape, bool)
                for k in keys:
                    a, b, c, d = self.pin_rects()[k]
                    m[b:d, a:c] = True
                masks.append((cls, m))
        self.draw_base(missing)
        for cls, size in specs:
            if cls == "crack":
                masks.append((cls, self.crack()))
            elif cls == "scratch":
                masks.append((cls, self.scratch(size, ppm)))
            elif cls == "contamination":
                masks.append((cls, self.contamination(size, ppm)))
            elif cls == "chip":
                masks.append((cls, self.chip(size, ppm)))
        # later defects may paint over earlier ones; keep masks consistent with what is visible
        cleaned, covered = [], np.zeros(self.body.shape, bool)
        for cls, m in reversed(masks):
            m = m & ~covered
            covered |= m
            if m.sum() >= 3:
                cleaned.append((cls, m))
        return list(reversed(cleaned))


# ---------------------------------------------------------- defect sampling

def sample_defects(rng) -> list[tuple[str, float]]:
    """Choose which defects a device gets, with sizes spanning the rule thresholds."""
    u = rng.random()
    if u < 0.30:
        return []                                         # clean device
    if u < 0.42:                                          # cosmetic only (should PASS)
        return [("scratch", rng.uniform(0.6, 1.3))] if rng.random() < 0.5 else \
               [("contamination", rng.uniform(0.15, 0.45))]
    specs = []
    for _ in range(rng.choice([1, 1, 1, 1, 1, 2, 2, 3])):
        cls = rng.choice(C.DEFECT_CLASSES, p=[0.11, 0.33, 0.33, 0.12, 0.11])
        size = {"crack": 0.0,
                "scratch": rng.uniform(1.8, 3.8) if rng.random() < 0.55 else rng.uniform(4.3, 9.0),
                "contamination": rng.uniform(0.7, 2.6) if rng.random() < 0.6 else rng.uniform(3.5, 7.0),
                "chip": rng.uniform(0.1, 1.8),
                "missing_pin": float(rng.choice([1, 1, 2]))}[cls]
        specs.append((cls, size))
    return specs


# -------------------------------------------------------------- the scene

PALETTES = [
    ((62, 78, 64), "green conveyor"),
    ((88, 70, 48), "blue tray"),
    ((40, 40, 44), "black ESD mat"),
    ((105, 108, 110), "grey belt"),
]


def _background(rng, size):
    colour, _ = PALETTES[rng.integers(0, len(PALETTES))]
    img = np.empty((size, size, 3), np.float32)
    jitter = rng.uniform(-10, 10, 3)
    for ch in range(3):
        img[..., ch] = colour[ch] + jitter[ch]
    tex = _texture(rng, img.shape, 20, 6) + _texture(rng, img.shape, 1.0, 3)
    img += tex[..., None]
    if rng.random() < 0.6:  # faint belt/tray lines
        step = int(rng.integers(40, 90))
        off = int(rng.integers(0, step))
        horizontal = rng.random() < 0.5
        for p in range(off, size, step):
            if horizontal:
                img[p:p + 2] *= 0.85
            else:
                img[:, p:p + 2] *= 0.85
    return img


def _rotate(canvas: _DeviceCanvas, masks, angle):
    S = canvas.S
    R = int(np.ceil(S * 1.45))
    pad = (R - S) // 2
    M = cv2.getRotationMatrix2D((R / 2, R / 2), angle, 1.0)

    def warp(a, interp):
        big = np.zeros((R, R) + a.shape[2:], a.dtype)
        big[pad:pad + S, pad:pad + S] = a
        return cv2.warpAffine(big, M, (R, R), flags=interp, borderValue=0)

    img = warp(canvas.img, cv2.INTER_LINEAR)
    alpha = warp(canvas.alpha.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
    out = []
    for cls, m in masks:
        w = warp(m.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
        if cls not in ("missing_pin", "chip"):   # these may lie where the device is now absent
            w &= alpha
        out.append((cls, w))
    foot = alpha.copy()
    for _, m in out:
        foot |= m
    ys, xs = np.nonzero(foot)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y1:y2, x1:x2], alpha[y1:y2, x1:x2], [(c, m[y1:y2, x1:x2]) for c, m in out]


def render_scene(rng, size: int = C.SCENE_SIZE, ppm: float = C.PX_PER_MM,
                 n_devices: int | None = None) -> Scene:
    img = _background(rng, size)
    occupied = np.zeros((size, size), bool)
    devices: list[Device] = []
    n = n_devices or int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
    for _ in range(n):
        canvas = _DeviceCanvas(rng, ppm)
        masks = canvas.render(ppm, sample_defects(rng))
        angle = float(rng.uniform(-15, 15) + 90 * rng.integers(0, 4))
        dimg, dalpha, dmasks = _rotate(canvas, masks, angle)
        h, w = dalpha.shape
        if h >= size - 8 or w >= size - 8:
            continue
        foot = dalpha.copy()
        for _, m in dmasks:
            foot |= m
        placed = False
        for _ in range(60):
            y, x = int(rng.integers(4, size - h - 4)), int(rng.integers(4, size - w - 4))
            region = occupied[max(0, y - 8):y + h + 8, max(0, x - 8):x + w + 8]
            if not region.any():
                placed = True
                break
        if not placed:
            continue
        # soft drop shadow, then paste the device
        shadow = np.zeros((size, size), np.float32)
        sy, sx = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        shadow[y + sy:y + sy + h, x + sx:x + sx + w] = dalpha[: size - y - sy, : size - x - sx]
        shadow = cv2.GaussianBlur(shadow, (0, 0), 2.5)
        img *= (1 - 0.35 * shadow)[..., None]
        view = img[y:y + h, x:x + w]
        view[dalpha] = dimg[dalpha]
        occupied[y:y + h, x:x + w] |= foot

        full = lambda m: np.pad(m, ((y, size - y - h), (x, size - x - w)))  # noqa: E731
        fp = full(foot)
        ys, xs = np.nonzero(fp)
        dev = Device(box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                     footprint=fp, body_mm=canvas.body_mm, angle=angle,
                     defects=[Defect(str(c), full(m)) for c, m in dmasks])
        devices.append(dev)

    # camera effects: uneven lighting, colour cast, sensor noise, slight defocus
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    a = rng.uniform(0, 2 * np.pi)
    light = 1 + rng.uniform(0.05, 0.22) * ((xx - 0.5) * np.cos(a) + (yy - 0.5) * np.sin(a))
    vignette = 1 - rng.uniform(0.0, 0.18) * (((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 2)
    img = img * (light * vignette)[..., None] * rng.uniform(0.85, 1.15)
    img += rng.uniform(-6, 6, 3)
    img += rng.normal(0, rng.uniform(1.5, 5.0), img.shape)
    s = rng.uniform(0, 0.8)
    if s > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), s)
    return Scene(np.clip(img, 0, 255).astype(np.uint8), devices)


# --------------------------------------------------------- ground truth

def device_ground_truth(dev: Device, rules: dict, ppm: float = C.PX_PER_MM) -> dict:
    defects = []
    for d in dev.defects:
        m = measure_mask(d.mask, ppm)
        verdict, reason = judge_defect(d.cls, m, rules)
        defects.append({"class": str(d.cls), "area_mm2": round(m.area_mm2, 4),
                        "length_mm": round(m.length_mm, 4), "verdict": verdict, "reason": reason})
    return {"box": list(dev.box), "body_mm": round(dev.body_mm, 2), "angle": round(dev.angle, 1),
            "decision": combine(d["verdict"] for d in defects), "defects": defects}
