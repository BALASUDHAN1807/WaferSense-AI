"""One entry point for every stage of the Semiconductor Device QC project.

    python run_project.py generate            build the synthetic dataset (750 frames, ~1460 devices)
    python run_project.py validate            check images, labels, masks and splits
    python run_project.py eda                 statistics + graphs in results/graphs/
    python run_project.py train-detector      YOLO11n device detector
    python run_project.py train-yoloseg       YOLO11n-seg defect segmenter
    python run_project.py train-unet          U-Net defect segmenter
    python run_project.py train-all           all three models
    python run_project.py evaluate            test-set metrics, end-to-end decisions, best pipeline
    python run_project.py inspect IMAGE       inspect one image (prints decisions, saves annotated image)
    python run_project.py camera [--source 0 | video.mp4] [--no-display] [--make-demo-video]
    python run_project.py api                 FastAPI backend   -> http://127.0.0.1:8000/docs
    python run_project.py dashboard           Streamlit dashboard -> http://localhost:8501
    python run_project.py serve               API + dashboard together
    python run_project.py finetune-yoloseg    2nd training stage: warm start + class oversampling + cosine LR
    python run_project.py real-eval           REAL images: MVTec AD transistor (PatchCore + YOLO11-seg CV)
    python run_project.py real-typecls        REAL images: defect-type classifier (full photo vs PatchCore crop)
    python run_project.py report              regenerate report/Project_Report.md and README results
    python run_project.py test                pytest suite
    python run_project.py all                 generate -> validate -> eda -> train-all -> evaluate -> report

Extra arguments after the command are passed through (e.g. --epochs 5).
Set QC_PROFILE=large (and QC_DEVICE=0 on a GPU) for YOLO11s-seg / 384 px / 100 epochs,
see notebooks/train_large_on_colab.ipynb.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))


def _py(*args) -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.call([sys.executable, *args], cwd=ROOT, env=env)


def _need(path: Path, hint: str) -> bool:
    if not path.exists():
        print(f"Missing {path.relative_to(ROOT)}. {hint}")
        return False
    return True


def cmd_generate(a):
    from qc.build_dataset import main
    return main(a)


def cmd_validate(a):
    from qc.eda import validate
    return 0 if validate()["passed"] else 1


def cmd_eda(a):
    from qc.eda import run_eda
    run_eda()
    return 0


def _train(model):
    def f(a):
        from qc import config as C
        if not _need(C.SCENE_DIR / "data.yaml", "Run: python run_project.py generate"):
            return 1
        from qc.train import main
        return main([model, *a])
    return f


def cmd_evaluate(a):
    from qc import config as C
    if not _need(C.DETECTOR_WEIGHTS, "Run: python run_project.py train-all"):
        return 1
    from qc.evaluate import main
    return main(a)


def cmd_inspect(a):
    if not a:
        print("Usage: python run_project.py inspect path/to/image.png [--segmenter yolo|unet]")
        return 1
    from qc.engine import main
    return main(a)


def cmd_camera(a):
    from qc.camera import main
    return main(a)


def cmd_api(a):
    return _py("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", *a)


def cmd_dashboard(a):
    return _py("-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true",
               "--browser.gatherUsageStats", "false", *a)


def cmd_serve(a):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                            "--port", "8000"], cwd=ROOT, env=env)
    print("API running at http://127.0.0.1:8000 (docs: /docs)")
    try:
        return cmd_dashboard(a)
    finally:
        api.terminate()


def cmd_real_eval(a):
    from qc.real.evaluate_real import main
    return main(a)


def cmd_finetune(a):
    from qc.finetune import main
    return main(a)


def cmd_real_typecls(a):
    from qc.real.type_classifier import main
    return main(a)


def cmd_report(a):
    from qc.report import main
    return main()


def cmd_test(a):
    return _py("-m", "pytest", *a)


def cmd_all(a):
    steps = [("generate", cmd_generate), ("validate", cmd_validate), ("eda", cmd_eda),
             ("train-detector", _train("detector")), ("train-yoloseg", _train("yolo-seg")),
             ("train-unet", _train("unet")), ("evaluate", cmd_evaluate), ("report", cmd_report)]
    for name, fn in steps:
        print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}")
        rc = fn([])
        if rc:
            print(f"Stage '{name}' failed (exit code {rc}). Stopping.")
            return rc
    print("\nAll stages completed. Start the system with: python run_project.py serve")
    return 0


COMMANDS = {
    "generate": cmd_generate, "validate": cmd_validate, "eda": cmd_eda,
    "train-detector": _train("detector"), "train-yoloseg": _train("yolo-seg"), "train-unet": _train("unet"),
    "train-all": _train("all"), "evaluate": cmd_evaluate, "inspect": cmd_inspect, "camera": cmd_camera,
    "api": cmd_api, "dashboard": cmd_dashboard, "serve": cmd_serve, "report": cmd_report, "real-eval": cmd_real_eval,
    "finetune-yoloseg": cmd_finetune, "real-typecls": cmd_real_typecls,
    "test": cmd_test, "all": cmd_all,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'.\n{__doc__}")
        return 2
    return COMMANDS[cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
