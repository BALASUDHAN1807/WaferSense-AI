"""U-Net for semantic defect segmentation (PyTorch).

Input:  3 x 256 x 256 device crop (BGR, scaled to [0, 1])
Output: 6 x 256 x 256 logits (0 = background, 1..5 = defect classes)

Classic encoder-decoder with skip connections (Ronneberger et al., 2015),
4 down-sampling levels, BatchNorm, base width configurable.
"""
from __future__ import annotations

import torch
from torch import nn

from qc import config as C

N_SEG_CLASSES = len(C.DEFECT_CLASSES) + 1


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = N_SEG_CLASSES, base: int = 16):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in ch[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(ch[-2], ch[-1])
        self.ups, self.up_convs = nn.ModuleList(), nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.up_convs.append(DoubleConv(c * 2, c))
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def to_tensor(crop_bgr) -> torch.Tensor:
    """uint8 HxWx3 (or NxHxWx3) -> float tensor NCHW in [0, 1]."""
    t = torch.as_tensor(crop_bgr).float() / 255.0
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.permute(0, 3, 1, 2).contiguous()


def load_unet(path=C.UNET_WEIGHTS) -> UNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(base=ckpt.get("base", C.UNET["base_channels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
