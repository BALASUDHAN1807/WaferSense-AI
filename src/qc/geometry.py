"""Device crop geometry shared by dataset building and inference.

A detected device box is turned into a square, padded crop resized to
CROP_SIZE x CROP_SIZE.  `CropInfo` keeps what is needed to map crop pixels
back to camera pixels (and therefore to millimetres).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qc import config as C


@dataclass
class CropInfo:
    x0: int          # top-left of the square crop in camera pixels (may be negative)
    y0: int
    side: int        # side length of the square crop in camera pixels
    size: int        # crop resolution after resizing

    @property
    def scale(self) -> float:
        """Camera pixels per crop pixel."""
        return self.side / self.size

    def to_camera(self, mask: np.ndarray, shape) -> np.ndarray:
        """Paste a crop-resolution mask back into a camera-resolution mask."""
        H, W = shape[:2]
        big = cv2.resize(mask.astype(np.uint8), (self.side, self.side), interpolation=cv2.INTER_NEAREST)
        out = np.zeros((H, W), bool)
        ys, xs = max(0, self.y0), max(0, self.x0)
        ye, xe = min(H, self.y0 + self.side), min(W, self.x0 + self.side)
        out[ys:ye, xs:xe] = big[ys - self.y0:ye - self.y0, xs - self.x0:xe - self.x0].astype(bool)
        return out


def square_crop(image: np.ndarray, box, pad: float = C.CROP_PAD, size: int = C.CROP_SIZE,
                jitter: np.ndarray | None = None):
    """Return (crop, CropInfo).  `jitter` = (dx, dy, dscale) fractions for training augmentation."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    if jitter is not None:
        cx += jitter[0] * side
        cy += jitter[1] * side
        side *= 1 + jitter[2]
    side = max(8, int(round(side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = image.shape[:2]
    # pad with edge-replicated borders when the crop leaves the frame
    padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
    crop = padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)
    return crop, CropInfo(x0, y0, side, size)


def crop_mask(mask: np.ndarray, info: CropInfo) -> np.ndarray:
    """Cut the same square region out of a camera-resolution mask (nearest neighbour)."""
    s = info.side
    padded = np.pad(mask.astype(np.uint8), s)
    m = padded[info.y0 + s:info.y0 + 2 * s, info.x0 + s:info.x0 + 2 * s]
    return cv2.resize(m, (info.size, info.size), interpolation=cv2.INTER_NEAREST).astype(bool)
