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
import zipfile
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


def _search(base: Path) -> Path | None:
    """Find a folder that looks like MVTec's transistor category anywhere below `base`."""
    for train_good in sorted(base.rglob("train/good")):
        root = train_good.parent.parent
        if (root / "test").exists() and (root / "ground_truth").exists():
            return root
    return None


def _safe_members(names):
    for n in names:
        if n.startswith(("/", "\\")) or ".." in Path(n).parts:
            raise ValueError(f"unsafe path in archive: {n}")


def find_root(base: Path = REAL_DIR) -> Path:
    """Return the transistor folder (MVTec layout), extracting a .tar.xz / .zip in `base` if needed.
    Works with the official MVTec archive and with Kaggle / mirror zips of the same files."""
    base.mkdir(parents=True, exist_ok=True)
    root = _search(base)
    if root:
        return root
    archives = sorted(base.glob("*.tar*")) + sorted(base.glob("*.zip"))
    for arc in archives:
        print(f"Extracting {arc.name} ...")
        dest = base / arc.name.split(".")[0]
        if zipfile.is_zipfile(arc):
            with zipfile.ZipFile(arc) as zf:
                _safe_members(zf.namelist())
                zf.extractall(dest)
        else:
            with tarfile.open(arc) as tf:
                _safe_members([m.name for m in tf.getmembers()])
                tf.extractall(dest)
        root = _search(base)
        if root:
            return root
    raise FileNotFoundError(
        f"MVTec AD '{CATEGORY}' not found in {base}.\n"
        "Put transistor.tar.xz (MVTec) or the Kaggle zip of the MVTec transistor category in dataset/real/ "
        "(it must contain train/good, test/<type> and ground_truth/<type>).")


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
