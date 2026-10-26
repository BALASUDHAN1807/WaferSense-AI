"""PatchCore anomaly detection on REAL images (MVTec AD transistor).

PatchCore (Roth et al., CVPR 2022) learns only from DEFECT-FREE images: it stores
mid-level CNN features of every image patch in a memory bank; at test time a patch
that is far from every stored patch is anomalous. It needs no defect labels, which
is how real factories start (defects are rare and unlabelled).

Backbone: ImageNet-pretrained YOLO11s-cls (Ultralytics). Features from two
intermediate stages (stride 8 and 16) are upsampled, concatenated and locally
averaged (3x3); a greedy coreset keeps 1 % of the patches.

Outputs per image: anomaly score (max patch distance) and a pixel anomaly map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qc import config as C

IMG_SIZE = 256
FEATURE_LAYERS = (4, 6)        # YOLO11s-cls stages with stride 8 and 16
CORESET_FRACTION = 0.01
PROJ_DIM = 128


def load_image(path, size: int = IMG_SIZE) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path, size: int = IMG_SIZE) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127


class Backbone(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        from ultralytics import YOLO
        weights = weights or (C.MODEL_DIR / "pretrained" / "yolo11s-cls.pt")
        layers = YOLO(str(weights)).model.model
        self.layers = torch.nn.ModuleList(list(layers)[: max(FEATURE_LAYERS) + 1]).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in FEATURE_LAYERS:
                feats.append(x)
        size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats]
        f = torch.cat(feats, 1)
        return F.avg_pool2d(f, 3, stride=1, padding=1)          # local neighbourhood aggregation


def greedy_coreset(feats: torch.Tensor, n: int, seed: int = C.SEED) -> torch.Tensor:
    """k-center greedy selection on a random projection (PatchCore's coreset)."""
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(feats.shape[1], PROJ_DIM, generator=g) / PROJ_DIM ** 0.5
    z = feats @ proj
    idx = [int(torch.randint(len(z), (1,), generator=g))]
    dmin = torch.cdist(z, z[idx[-1]][None]).squeeze(1)
    for _ in range(n - 1):
        i = int(dmin.argmax())
        idx.append(i)
        dmin = torch.minimum(dmin, torch.cdist(z, z[i][None]).squeeze(1))
    return feats[idx]


class PatchCore:
    def __init__(self, backbone: Backbone | None = None):
        self.backbone = backbone or Backbone()
        self.bank: torch.Tensor | None = None
        self.fmap = None

    def _features(self, paths, batch: int = 8):
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack([load_image(p) for p in paths[i:i + batch]])
            out.append(self.backbone(x))
        return torch.cat(out)

    def fit(self, good_paths):
        f = self._features(good_paths)                       # N, C, H, W
        self.fmap = f.shape[-2:]
        patches = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        n = max(1000, int(len(patches) * CORESET_FRACTION))
        self.bank = greedy_coreset(patches, n)
        return self

    @torch.no_grad()
    def score(self, paths, batch: int = 8):
        """Return (image_scores [N], anomaly_maps [N, IMG_SIZE, IMG_SIZE])."""
        scores, maps = [], []
        for i in range(0, len(paths), batch):
            f = self._features(paths[i:i + batch])
            B, Cf, H, W = f.shape
            q = f.permute(0, 2, 3, 1).reshape(-1, Cf)
            d = torch.cat([torch.cdist(q[j:j + 4096], self.bank).min(1).values
                           for j in range(0, len(q), 4096)])
            m = d.reshape(B, 1, H, W)
            m = F.interpolate(m, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            m = _gaussian_blur(m, sigma=4)
            maps.append(m[:, 0].numpy())
            scores.append(d.reshape(B, -1).max(1).values.numpy())
        return np.concatenate(scores), np.concatenate(maps)

    def save(self, path):
        torch.save({"bank": self.bank, "fmap": tuple(self.fmap), "img_size": IMG_SIZE,
                    "layers": FEATURE_LAYERS}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.bank, self.fmap = ck["bank"], ck["fmap"]
        return self


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(4 * sigma + 1) | 1
    t = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
