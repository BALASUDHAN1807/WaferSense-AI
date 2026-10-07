"""One entry point for every stage of the project.

    python run_project.py generate             create the synthetic dataset (900 images)
    python run_project.py validate             validate dataset/labels.csv and images
    python run_project.py eda                  exploratory data analysis + graphs
    python run_project.py preprocess           create/verify the 70/15/15 split
    python run_project.py models               build all 3 models and check output shapes
    python run_project.py train-cnn
    python run_project.py train-resnet
    python run_project.py train-efficientnet
    python run_project.py train-all            train all 3 models
    python run_project.py evaluate             test-set evaluation of trained models
    python run_project.py compare              comparison table + best-model selection
    python run_project.py predict IMAGE        predict one wafer image with the best model
    python run_project.py summary              write results/reports/final_project_summary.json
    python run_project.py report               write report/Project_Report.md + README results
    python run_project.py test                 run the pytest suite
    python run_project.py app                  start the Gradio web app
    python run_project.py all                  validate -> eda -> preprocess -> train-all
                                               -> evaluate -> compare -> summary -> report

Extra arguments after the command are passed through (e.g. --epochs 5).
Evaluation never retrains a model.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _run(module: str, args=None) -> int:
    import inspect
    fn = getattr(__import__(module), "main")
    takes_args = len(inspect.signature(fn).parameters) > 0
    return fn(args or []) if takes_args else fn()


def _need(path: Path, hint: str) -> bool:
    if not path.exists():
        print(f"Missing {path.relative_to(ROOT)}. {hint}")
        return False
    return True


def cmd_generate(args):
    return _run("dataset", ["--generate", *args])


def cmd_train(name):
    def _f(args):
        import config as C
        if not _need(C.LABEL_FILE, "Run: python run_project.py generate"):
            return 1
        return _run({"CNN": "train_cnn", "ResNet50": "train_resnet",
                     "EfficientNetB0": "train_efficientnet"}[name], args)
    return _f


def cmd_train_all(args):
    for name in ("CNN", "ResNet50", "EfficientNetB0"):
        rc = cmd_train(name)(args)
        if rc:
            return rc
    return 0


def cmd_evaluate(args):
    import config as C
    if not any((C.MODEL_DIR / m["file"]).exists() for m in C.MODELS.values()):
        print("No trained models in models/. Run: python run_project.py train-all")
        return 1
    return _run("evaluate", args)


def cmd_compare(args):
    import config as C
    if not _need(C.REPORT_DIR / "all_model_results.json", "Run: python run_project.py evaluate"):
        return 1
    return _run("compare_models")


def cmd_predict(args):
    if not args:
        print("Usage: python run_project.py predict path/to/wafer.png [--model CNN]")
        return 1
    return _run("predict", args)


def cmd_summary(args):
    from summary import main
    return main()


def cmd_test(args):
    return subprocess.call([sys.executable, "-m", "pytest", *args], cwd=ROOT)


def cmd_app(args):
    return subprocess.call([sys.executable, str(ROOT / "app" / "app.py")], cwd=ROOT)


def cmd_all(args):
    import config as C
    steps = []
    if not C.LABEL_FILE.exists():
        steps.append(("generate", lambda: cmd_generate([])))
    steps += [
        ("validate", lambda: _run("validate_dataset")),
        ("eda", lambda: _run("eda")),
        ("preprocess", lambda: _run("preprocessing")),
        ("train-all", lambda: cmd_train_all(args)),
        ("evaluate", lambda: cmd_evaluate([])),
        ("compare", lambda: cmd_compare([])),
        ("summary", lambda: cmd_summary([])),
        ("report", lambda: _run("make_report")),
    ]
    for name, fn in steps:
        print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}")
        rc = fn()
        if rc:
            print(f"Stage '{name}' failed (exit code {rc}). Stopping.")
            return rc
    print("\nAll stages completed. Start the app with: python run_project.py app")
    return 0


COMMANDS = {
    "generate": cmd_generate,
    "validate": lambda a: _run("validate_dataset"),
    "eda": lambda a: _run("eda"),
    "preprocess": lambda a: _run("preprocessing"),
    "models": lambda a: _run("models"),
    "train-cnn": cmd_train("CNN"),
    "train-resnet": cmd_train("ResNet50"),
    "train-efficientnet": cmd_train("EfficientNetB0"),
    "train-all": cmd_train_all,
    "evaluate": cmd_evaluate,
    "compare": cmd_compare,
    "predict": cmd_predict,
    "summary": cmd_summary,
    "report": lambda a: _run("make_report"),
    "test": cmd_test,
    "app": cmd_app,
    "all": cmd_all,
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
