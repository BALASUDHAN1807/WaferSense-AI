"""MVTec AD loader (real industrial images).

MVTec AD (Bergmann et al., CVPR 2019) is a public dataset of REAL photographs of
industrial products with pixel-accurate defect masks. We use the `transistor`
category: through-hole transistors photographed from above, the closest category
to semiconductor devices.

    transistor/train/good/*.png                 defect-free images (training)
    transistor/test/<type>/*.png                test images: good, bent_lead, cut_lead,
                                                damaged_case, misplaced
    transistor/ground_truth/<type>/*_mask.png   pixel masks of every defect

Download (free for non-commercial / academic use, CC BY-NC-SA 4.0):
    https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads  -> transistor.tar.xz
Put the archive (or the extracted `transistor` folder) in dataset/real/.
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qc import config as C

REAL_DIR = C.DATASET_DIR / "real"
CATEGORY = "transistor"
DEFECT_TYPES = ["bent_lead", "cut_lead", "damaged_case", "misplaced"]


@dataclass
class Sample:
    path: Path
    label: str              # "good" or a defect type
    mask: Path | None       # ground-truth mask for defective images
    origin: str             # "train" or "test" (MVTec's own split)

    @property
    def defective(self) -> bool:
        return self.label != "good"


def find_root(base: Path = REAL_DIR) -> Path:
    """Return the extracted `transistor` folder, extracting the .tar.xz if needed."""
    for cand in (base / CATEGORY, base / "mvtec_anomaly_detection" / CATEGORY):
        if (cand / "train" / "good").exists():
            return cand
    archives = sorted(base.glob(f"{CATEGORY}*.tar*")) + sorted(base.glob("mvtec_anomaly_detection*.tar*"))
    if archives:
        print(f"Extracting {archives[0].name} ...")
        with tarfile.open(archives[0]) as tf:
            members = [m for m in tf.getmembers() if f"{CATEGORY}/" in m.name or m.name == CATEGORY]
            for m in members:   # refuse absolute paths / path traversal
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise ValueError(f"unsafe path in archive: {m.name}")
            tf.extractall(base, members=members)
        return find_root(base)
    raise FileNotFoundError(
        f"MVTec AD '{CATEGORY}' not found in {base}.\n"
        "Download transistor.tar.xz from https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads "
        "and put it in dataset/real/")


def load_samples(root: Path | None = None) -> list[Sample]:
    root = root or find_root()
    out = [Sample(p, "good", None, "train") for p in sorted((root / "train" / "good").glob("*.png"))]
    for d in sorted((root / "test").iterdir()):
        for p in sorted(d.glob("*.png")):
            mask = None
            if d.name != "good":
                mask = root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                if not mask.exists():
                    raise FileNotFoundError(f"missing mask {mask}")
            out.append(Sample(p, d.name, mask, "test"))
    return out


def split_test(samples: list[Sample], seed: int = C.SEED, calib_frac: float = 0.4):
    """Split MVTec's test images into a calibration half (threshold / hyper-parameter
    choice) and a held-out evaluation half, stratified by label. MVTec has no
    validation split, so this keeps threshold tuning away from the reported numbers."""
    rng = np.random.default_rng(seed)
    calib, evals = [], []
    test = [s for s in samples if s.origin == "test"]
    for label in sorted({s.label for s in test}):
        group = [s for s in test if s.label == label]
        idx = rng.permutation(len(group))
        k = int(round(len(group) * calib_frac))
        calib += [group[i] for i in idx[:k]]
        evals += [group[i] for i in idx[k:]]
    return calib, evals
