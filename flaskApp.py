import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import config
from models.cnn import FlowerCNN
from utils.imagePreprocess import preprocessUpload, preprocessUrl

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

_model:     FlowerCNN = None
_idxToName: dict      = None
_device:    torch.device = config.DEVICE


def loadModel():
   
    global _model, _idxToName

    if not config.LABEL_MAP_CACHE.exists():
        raise FileNotFoundError(
            f"Label map not found: {config.LABEL_MAP_CACHE}\n"
            f"Run 'python main.py --mode train' before starting the app."
        )
    with open(config.LABEL_MAP_CACHE) as f:
        raw = json.load(f)
    _idxToName = {int(k): v for k, v in raw.items()}

    if not config.CNN_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {config.CNN_CHECKPOINT}\n"
            f"Run 'python main.py --mode train' before starting the app."
        )

    checkpoint = torch.load(
        config.CNN_CHECKPOINT,
        map_location=_device,
        weights_only=False,
    )

    _model = FlowerCNN(
        numClasses=checkpoint.get("numClasses",    config.NUM_CLASSES),
        convChannels=checkpoint.get("convChannels", config.CONV_CHANNELS),
        fcHiddenDim=checkpoint.get("fcHiddenDim",   config.FC_HIDDEN_DIM),
        dropoutRate=checkpoint.get("dropoutRate",    config.DROPOUT_RATE),
    ).to(_device)

    _model.load_state_dict(checkpoint["modelStateDict"])
    _model.eval()

    epochInfo = checkpoint.get("epoch", "?")
    valAcc    = checkpoint.get("valAcc", 0) * 100
    print(f"  ✓ Model loaded  — epoch {epochInfo}, val acc {valAcc:.2f}%")
    print(f"  ✓ Label map     — {len(_idxToName)} classes")
    print(f"  ✓ Device        — {_device}")


@torch.no_grad()
def runInference(tensor: torch.Tensor) -> list[dict]:

    logits = _model(tensor)                          # (1, 102)
    probs  = F.softmax(logits, dim=1)[0]             # (102,)

    topK   = min(config.TOP_K_PREDICTIONS, len(_idxToName))
    values, indices = torch.topk(probs, k=topK)

    predictions = []
    for rank, (prob, idx) in enumerate(
        zip(values.cpu().tolist(), indices.cpu().tolist()), start=1
    ):
        predictions.append({
            "rank":       rank,
            "name":       _idxToName.get(idx, f"class_{idx}"),
            "classIndex": idx,
            "confidence": round(prob * 100, 2),      # percentage, 2 d.p.
        })

    return predictions


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if _model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Run training first.",
        }), 503

    try:
        inputType = None
        tensor    = None

        # ── Branch A: file upload ─────────────────────────────────────────────
        if "image" in request.files and request.files["image"].filename != "":
            fileObj  = request.files["image"]
            filename = secure_filename(fileObj.filename)
            ext      = Path(filename).suffix.lower().lstrip(".")

            if ext not in config.ALLOWED_EXTENSIONS:
                return jsonify({
                    "success": False,
                    "error": (
                        f"Unsupported file type: .{ext}. "
                        f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
                    ),
                }), 400

            tensor    = preprocessUpload(fileObj)
            inputType = "upload"

        elif "imageUrl" in request.form and request.form["imageUrl"].strip():
            url       = request.form["imageUrl"].strip()
            tensor    = preprocessUrl(url)
            inputType = "url"

        else:
            return jsonify({
                "success": False,
                "error": "No image provided. Upload a file or enter an image URL.",
            }), 400

        predictions = runInference(tensor)

        return jsonify({
            "success":     True,
            "predictions": predictions,
            "inputType":   inputType,
        }), 200

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error":   f"Server error: {type(e).__name__}: {str(e)}",
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight health check endpoint.
    Returns model status and class count — useful for debugging.
    """
    return jsonify({
        "status":      "ok",
        "modelLoaded": _model is not None,
        "numClasses":  len(_idxToName) if _idxToName else 0,
        "device":      str(_device),
    }), 200


def runApp():
    """Called by main.py --mode app."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   Flower Species Recognition — Flask App                     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"  Loading model from: {config.CNN_CHECKPOINT}")
    loadModel()
    print(f"  Starting server  : http://localhost:{config.FLASK_PORT}")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )


if __name__ == "__main__":
    runApp()