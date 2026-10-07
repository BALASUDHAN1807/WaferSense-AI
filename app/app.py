"""Gradio web application for wafer defect pattern detection.

Loads the best trained model once (from results/reports/best_model.json) and
lets the user upload a wafer-map image to get the predicted defect class,
confidence, confidence interpretation and the probability of all 9 classes.

Run from the project root:
    python app/app.py
Then open http://127.0.0.1:7860 in a browser.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

import config as C  # noqa: E402
from predict import WaferPredictor, best_model_info  # noqa: E402

DISCLAIMER = ("This system is an academic ML prototype. Predictions are based on the trained "
              "dataset and should not be treated as a direct manufacturing quality-control decision.")

# Load the model ONCE at start-up (never retrains).
PREDICTOR, LOAD_ERROR, MODEL_INFO = None, None, {}
try:
    PREDICTOR = WaferPredictor()
    MODEL_INFO = best_model_info()
except Exception as e:  # missing model / missing best_model.json
    LOAD_ERROR = str(e)


def _model_markdown() -> str:
    if LOAD_ERROR:
        return (f"**Model not loaded:** {LOAD_ERROR}\n\n"
                "Train and evaluate the models first: `python run_project.py all`")
    return (f"**Active model:** {MODEL_INFO['model_name']} (selected automatically)  \n"
            f"Test macro F1: **{MODEL_INFO['metric_value']:.4f}** · "
            f"Test accuracy: **{MODEL_INFO['accuracy']:.4f}** · "
            f"Inference: {MODEL_INFO['inference_time_ms_per_image']:.2f} ms/image  \n"
            f"<small>{MODEL_INFO['reason']}</small>")


def classify(image):
    """Gradio callback -> (class markdown, probabilities dict, status message)."""
    if PREDICTOR is None:
        return "### Model not available", None, f"Error: {LOAD_ERROR}"
    if image is None:
        return "### No image", None, "Please upload a wafer-map image first."
    try:
        r = PREDICTOR.predict(image)
    except Exception as e:
        return "### Prediction failed", None, f"Error: {e}"
    result_md = (
        f"## Predicted defect: {r['predicted_class']}\n"
        f"**Confidence:** {r['confidence_percent']:.2f}%  \n"
        f"**Interpretation:** {r['confidence_interpretation']}"
    )
    status = (f"Predicted with {r['model']}. {r['note']}")
    return result_md, r["all_class_probabilities"], status


def _examples() -> list:
    """One example image per class from the held-out test split (if available)."""
    try:
        import pandas as pd
        split = pd.read_csv(C.SPLIT_FILE, keep_default_na=False)
        test = split[split["split"] == "test"]
        return [[str(C.IMAGE_DIR / test[test["label"] == c]["image"].iloc[0])]
                for c in C.CLASS_NAMES if (test["label"] == c).any()]
    except Exception:
        return []


with gr.Blocks(title="Semiconductor Wafer Defect Detection AI") as demo:
    gr.Markdown(
        "# AI-Based Semiconductor Wafer Defect Pattern Detection\n"
        "Upload a **wafer-map image** (black = outside the wafer, grey = good die, "
        "white = defective die). The trained deep-learning model classifies the defect "
        "pattern into one of 9 classes: " + ", ".join(C.CLASS_NAMES) + "."
    )
    gr.Markdown(_model_markdown())
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Upload Wafer Map", image_mode="L", height=320)
            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.ClearButton(value="Clear")
            examples = _examples()
            if examples:
                gr.Examples(examples=examples, inputs=image_in,
                            label="Example wafer maps (one per class, from the test set)")
        with gr.Column(scale=1):
            result_out = gr.Markdown("### Result will appear here")
            probs_out = gr.Label(num_top_classes=C.NUM_CLASSES, label="Probability of all 9 classes")
            status_out = gr.Textbox(label="Status", interactive=False)
    gr.Markdown(f"> **Disclaimer:** {DISCLAIMER}")

    predict_btn.click(classify, inputs=image_in, outputs=[result_out, probs_out, status_out])
    clear_btn.add([image_in, result_out, probs_out, status_out])


if __name__ == "__main__":
    demo.launch()
