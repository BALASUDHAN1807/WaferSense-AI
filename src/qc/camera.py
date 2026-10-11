"""Real-time inspection from a camera or a video file (OpenCV).

    python run_project.py camera --source 0                 # webcam 0
    python run_project.py camera --source conveyor.mp4      # video file
    python run_project.py camera --make-demo-video          # build results/conveyor_demo.mp4 from test frames

Every Nth frame (--every) is inspected; results are drawn on the frame, shown
in a window (unless --no-display) and optionally written to --output and the
SQLite database (--log).  Press q to quit.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from qc import config as C


def make_demo_video(path=None, n_frames: int = 30, fps: float = 1.0) -> str:
    """Stitch held-out test frames into a 'conveyor' video (one product tray per second)."""
    path = str(path or C.RESULT_DIR / "conveyor_demo.mp4")
    frames = sorted((C.SCENE_DIR / "test" / "images").glob("*.png"))[:n_frames]
    if not frames:
        raise FileNotFoundError("No test frames - run: python run_project.py generate")
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(cv2.imread(str(f)))
    out.release()
    print(f"Demo video: {path} ({len(frames)} frames @ {fps} fps)")
    return path


def run(source, every: int = 1, display: bool = True, output: str | None = None, log: bool = False,
        segmenter: str | None = None, max_frames: int | None = None) -> dict:
    from qc import db
    from qc.engine import InspectionEngine, annotate

    engine = InspectionEngine(segmenter)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")
    writer, totals, k, t0 = None, {d: 0 for d in C.DECISIONS}, 0, time.perf_counter()
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k += 1
        if (k - 1) % every == 0:
            res = engine.inspect(frame)
            for d, v in res.counts.items():
                totals[d] += v
            last = res
            if log:
                db.save_inspection(res, source=f"camera:{source}#{k}")
        vis = annotate(frame, last) if last is not None else frame
        if output:
            if writer is None:
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 10, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if display:
            cv2.imshow("Semiconductor QC - press q to quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and k >= max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    secs = time.perf_counter() - t0
    summary = {"frames": k, "devices": totals, "seconds": round(secs, 1),
               "fps": round(k / secs, 2) if secs else None}
    print(f"Processed {k} frames in {secs:.1f} s ({summary['fps']} fps): {totals}")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live / video inspection")
    p.add_argument("--source", default="0", help="webcam index or video file")
    p.add_argument("--every", type=int, default=1, help="inspect every Nth frame")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--output", default=None, help="save annotated video here (.mp4)")
    p.add_argument("--log", action="store_true", help="store results in the database")
    p.add_argument("--segmenter", choices=["yolo", "unet"], default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--make-demo-video", action="store_true")
    a = p.parse_args(argv)
    if a.make_demo_video:
        make_demo_video()
        return 0
    try:
        run(a.source, a.every, not a.no_display, a.output, a.log, a.segmenter, a.max_frames)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
