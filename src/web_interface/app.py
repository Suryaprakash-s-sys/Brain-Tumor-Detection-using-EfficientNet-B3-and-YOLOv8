"""
Flask Web Interface for Brain Tumor Detection System
Provides REST API and HTML frontend for uploading MRI images
and receiving real-time detection/classification results.
"""

import os
import uuid
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from werkzeug.utils import secure_filename

import sys
sys.path.append(str(Path(__file__).parent.parent))

from detection.yolov8_detector import TumorDetector
from classification.efficientnet_classifier import TumorClassifier

# ── Config ──────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = Path("results")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

# ── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Load models (loaded once at startup)
detector = TumorDetector(weights=os.getenv("YOLO_WEIGHTS", "../../models/yolov8_brain_tumor.pt"))
classifier = TumorClassifier(weights=os.getenv("EFFNET_WEIGHTS", "../../models/efficientnet_b3_brain_tumor.pt"))

# MongoDB
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
db = MongoClient(mongo_uri).brain_tumor_db
results_collection = db.results


# ── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG or PNG."}), 400

    # Save uploaded image
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = UPLOAD_FOLDER / filename
    file.save(str(filepath))

    # Run detection
    detections = detector.predict(str(filepath))

    # Run classification on each detected region
    prediction_results = []
    for det in detections:
        result = {
            "tumor_detected": True,
            "bounding_box": det.bbox,
            "detection_confidence": det.confidence,
        }
        if det.cropped_image is not None:
            import tempfile, cv2
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, det.cropped_image)
                cls = classifier.predict(tmp.name)
            result["classification"] = cls
        prediction_results.append(result)

    if not detections:
        prediction_results = [{"tumor_detected": False, "message": "No tumor detected"}]

    # Save to MongoDB
    record = {
        "filename": filename,
        "timestamp": datetime.utcnow().isoformat(),
        "results": prediction_results,
    }
    results_collection.insert_one(record)

    return jsonify({"status": "success", "results": prediction_results})


@app.route("/history")
def history():
    """Return past analyses from MongoDB."""
    records = list(results_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
    return jsonify(records)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
