"""Training for the three models.

    detector    YOLO11n           camera frame -> IC device boxes
    yolo-seg    YOLO11n-seg       device crop  -> defect instance masks (5 classes)
    unet        U-Net (PyTorch)   device crop  -> defect semantic mask (5 classes + background)

Every run saves the final weights to models/ and a JSON summary to
results/reports/<model>_training.json.  The test split is never used here.

Usage:
    python run_project.py train-detector | train-yoloseg | train-unet | train-all
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request

import cv2
import numpy as np

from qc import config as C


def _save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def pretrained(name: str):
    """Path to a pretrained YOLO11 checkpoint, downloading it from GitHub if needed."""
    dest = C.MODEL_DIR / "pretrained" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(C.YOLO_ASSET_URL.format(name=name), dest)
    return dest


# ------------------------------------------------------------------ YOLO

def _train_yolo(kind: str, cfg: dict, data_yaml, out_weights, epochs=None, extra=None) -> dict:
    from ultralytics import YOLO

    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} missing - run: python run_project.py generate")
    C.ensure_dirs()
    model = YOLO(str(pretrained(cfg["base"])))
    t0 = time.perf_counter()
    model.train(data=str(data_yaml), imgsz=cfg["imgsz"], epochs=epochs or cfg["epochs"],
                batch=cfg["batch"], patience=cfg["patience"], device=C.DEVICE, workers=2,
                seed=C.SEED, deterministic=True, project=str(C.RUNS_DIR), name=kind, exist_ok=True,
                plots=True, verbose=False, **(extra or {}))
    seconds = time.perf_counter() - t0
    run_dir = C.RUNS_DIR / kind
    shutil.copy(run_dir / "weights" / "best.pt", out_weights)

    import pandas as pd
    hist = pd.read_csv(run_dir / "results.csv")
    hist.columns = [c.strip() for c in hist.columns]
    key = "metrics/mAP50-95(M)" if "metrics/mAP50-95(M)" in hist else "metrics/mAP50-95(B)"
    best = hist.loc[hist[key].idxmax()]
    summary = {
        "model": kind, "base_weights": cfg["base"], "weights": str(out_weights.relative_to(C.ROOT)),
        "imgsz": cfg["imgsz"], "batch": cfg["batch"], "epochs_configured": epochs or cfg["epochs"],
        "epochs_completed": int(len(hist)), "best_epoch": int(best["epoch"]),
        "training_time_seconds": round(seconds, 1),
        "best_val": {k.split("/")[-1]: round(float(best[k]), 4) for k in hist.columns if k.startswith("metrics/")},
        "device": C.DEVICE,
    }
    _save_json(summary, C.REPORT_DIR / f"{kind}_training.json")
    for f in ("results.png",):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, C.GRAPH_DIR / f"{kind}_training_curves.png")
    print(json.dumps(summary, indent=2))
    return summary


def train_detector(epochs=None):
    return _train_yolo("detector", C.DETECTOR, C.SCENE_DIR / "data.yaml", C.DETECTOR_WEIGHTS, epochs)


def train_yolo_seg(epochs=None):
    # crops are rotation-invariant: allow vertical flips too
    return _train_yolo("yolo_seg", C.YOLO_SEG, C.DEFECT_DIR / "data.yaml", C.YOLO_SEG_WEIGHTS, epochs,
                       extra={"flipud": 0.5, "mosaic": 0.5})


# ------------------------------------------------------------------ U-Net

class CropDataset:
    def __init__(self, split: str, augment: bool):
        d = C.DEFECT_DIR / split
        self.images = sorted((d / "images").glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No crops in {d} - run: python run_project.py generate")
        self.mask_dir = d / "masks"
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        import torch
        img = cv2.imread(str(self.images[i]))
        mask = cv2.imread(str(self.mask_dir / self.images[i].name), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            # train on 128x128 patches (4x cheaper on CPU); 70 % are centred on a defect pixel.
            # The U-Net is fully convolutional, so it still runs on full 256x256 crops later.
            P = C.UNET["patch"]
            ys, xs = np.nonzero(mask)
            if len(ys) and np.random.rand() < 0.7:
                j = np.random.randint(len(ys))
                cy, cx = ys[j] + np.random.randint(-P // 4, P // 4), xs[j] + np.random.randint(-P // 4, P // 4)
            else:
                cy, cx = np.random.randint(P // 2, mask.shape[0] - P // 2, 2)
            y0 = int(np.clip(cy - P // 2, 0, mask.shape[0] - P))
            x0 = int(np.clip(cx - P // 2, 0, mask.shape[1] - P))
            img, mask = img[y0:y0 + P, x0:x0 + P], mask[y0:y0 + P, x0:x0 + P]
            k = np.random.randint(4)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.85, 1.15)
                          + np.random.uniform(-12, 12), 0, 255)
        img = torch.as_tensor(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
        return img, torch.as_tensor(np.ascontiguousarray(mask)).long()


def _confusion(pred, target, n):
    import torch
    idx = target.flatten() * n + pred.flatten()
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def iou_dice_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
    return iou, dice


def train_unet(epochs=None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from qc.unet import N_SEG_CLASSES, UNet

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    C.ensure_dirs()
    cfg = C.UNET
    epochs = epochs or cfg["epochs"]
    train_dl = DataLoader(CropDataset("train", True), batch_size=cfg["batch"], shuffle=True, num_workers=0)
    val_dl = DataLoader(CropDataset("val", False), batch_size=cfg["batch"], shuffle=False, num_workers=0)

    # class weights: inverse sqrt frequency (background is ~99 % of pixels)
    counts = np.zeros(N_SEG_CLASSES)
    for p in CropDataset("train", False).images:
        m = cv2.imread(str(C.DEFECT_DIR / "train" / "masks" / p.name), cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(m.ravel(), minlength=N_SEG_CLASSES)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    weights = weights / weights[0].item()

    model = UNet(base=cfg["base_channels"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    ce = torch.nn.CrossEntropyLoss(weight=weights)

    def dice_loss(logits, target):
        prob = logits.softmax(1)[:, 1:]
        onehot = torch.nn.functional.one_hot(target, N_SEG_CLASSES).permute(0, 3, 1, 2)[:, 1:].float()
        inter = (prob * onehot).sum((0, 2, 3))
        denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        return 1 - ((2 * inter + 1) / (denom + 1)).mean()

    history, best, best_epoch, wait = [], -1.0, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y) + dice_loss(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(x)
            n += len(x)
        model.eval()
        cm = torch.zeros(N_SEG_CLASSES, N_SEG_CLASSES, dtype=torch.long)
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                out = model(x)
                vl += (ce(out, y) + dice_loss(out, y)).item() * len(x)
                vn += len(x)
                cm += _confusion(out.argmax(1), y, N_SEG_CLASSES)
        iou, dice = iou_dice_from_confusion(cm.numpy())
        miou = float(np.nanmean(iou[1:]))
        history.append({"epoch": epoch, "train_loss": tl / n, "val_loss": vl / vn, "val_defect_mIoU": miou,
                        "val_defect_mDice": float(np.nanmean(dice[1:])), "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch:3d}  train_loss {tl / n:.4f}  val_loss {vl / vn:.4f}  "
              f"val defect mIoU {miou:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        sched.step(miou)
        if miou > best:
            best, best_epoch, wait = miou, epoch, 0
            torch.save({"state_dict": model.state_dict(), "base": cfg["base_channels"],
                        "classes": ["background"] + C.DEFECT_CLASSES}, C.UNET_WEIGHTS)
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"Early stopping (no improvement for {wait} epochs)")
                break
    seconds = time.perf_counter() - t0

    _plot_unet_history(history)
    summary = {"model": "unet", "weights": str(C.UNET_WEIGHTS.relative_to(C.ROOT)),
               "parameters": int(sum(p.numel() for p in model.parameters())),
               "base_channels": cfg["base_channels"], "batch": cfg["batch"], "lr": cfg["lr"],
               "loss": "weighted cross-entropy + soft Dice", "epochs_configured": epochs,
               "epochs_completed": len(history), "best_epoch": best_epoch,
               "best_val_defect_mIoU": round(best, 4), "training_time_seconds": round(seconds, 1),
               "device": C.DEVICE, "history": history}
    _save_json(summary, C.REPORT_DIR / "unet_training.json")
    return summary


def _plot_unet_history(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set_title("U-Net loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(ep, [h["val_defect_mIoU"] for h in history], label="mIoU")
    ax[1].plot(ep, [h["val_defect_mDice"] for h in history], label="mDice")
    ax[1].set_title("U-Net validation (defect classes)")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(C.GRAPH_DIR / "unet_training_curves.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the QC models")
    p.add_argument("model", choices=["detector", "yolo-seg", "unet", "all"])
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args(argv)
    jobs = {"detector": train_detector, "yolo-seg": train_yolo_seg, "unet": train_unet}
    for name in (jobs if a.model == "all" else [a.model]):
        print(f"\n=== Training {name} ===")
        jobs[name](a.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
