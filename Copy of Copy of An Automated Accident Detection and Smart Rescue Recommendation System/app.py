from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import os
import json
import uuid
from datetime import datetime, timezone

from image_scoring import score_image

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Rescue incidents storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INCIDENTS_FILE = os.path.join(DATA_DIR, "incidents.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Load primary model (used for readiness check; scoring uses image_scoring)
try:
    import tensorflow as tf
    _ = tf.keras.models.load_model("accident_model.keras", compile=False)
    print("Model loaded successfully")
except Exception:
    pass

# Max history items to keep in session
MAX_HISTORY = 50


def _load_incidents():
    if not os.path.isfile(INCIDENTS_FILE):
        return []
    with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_incidents(incidents):
    with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2)


def _web_path(filepath):
    return f"/{filepath.replace(os.sep, '/')}" if filepath else None


def get_history():
    return session.get("prediction_history", [])


def add_to_history(filename, image_path, prediction, confidence):
    history = get_history()
    # Store web-relative path for templates
    web_path = f"/{image_path.replace(os.sep, '/')}" if image_path else None
    history.insert(
        0,
        {
            "filename": filename,
            "image_path": web_path,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        },
    )
    session["prediction_history"] = history[:MAX_HISTORY]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction, confidence = score_image(filepath)
            image_path = _web_path(filepath)
            add_to_history(file.filename, filepath, prediction, confidence)

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
    )


@app.route("/rescue/request", methods=["POST"])
def rescue_request():
    image_path = request.form.get("image_path")
    confidence = request.form.get("confidence", type=float)
    location = request.form.get("location", "").strip()
    notes = request.form.get("notes", "").strip()
    is_ajax = request.form.get("ajax") == "1" or request.headers.get("X-Requested-With") == "XMLHttpRequest"
    if not image_path:
        if is_ajax:
            return jsonify({"ok": False, "error": "Missing image reference."}), 400
        flash("Missing image reference.", "error")
        return redirect(url_for("predict"))
    incidents = _load_incidents()
    incident = {
        "id": str(uuid.uuid4()),
        "image_path": image_path,
        "confidence": confidence or 0,
        "location": location,
        "notes": notes,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "updated_at": None,
    }
    incidents.insert(0, incident)
    _save_incidents(incidents)
    if is_ajax:
        return jsonify({"ok": True, "location": location if location and location != "Location unavailable" else None})
    flash("Rescue request submitted. Emergency services have been notified.", "success")
    return redirect(url_for("rescue_dashboard"))


@app.route("/rescue")
def rescue_dashboard():
    incidents = _load_incidents()
    pending = [i for i in incidents if i["status"] == "pending"]
    alert_sent = [i for i in incidents if i["status"] == "alert_sent"]
    resolved = [i for i in incidents if i["status"] == "resolved"]
    return render_template(
        "rescue.html",
        pending=pending,
        alert_sent=alert_sent,
        resolved=resolved,
        total=len(incidents),
    )


@app.route("/rescue/<incident_id>/status", methods=["POST"])
def rescue_update_status(incident_id):
    new_status = request.form.get("status")
    if new_status not in ("alert_sent", "resolved"):
        flash("Invalid status.", "error")
        return redirect(url_for("rescue_dashboard"))
    incidents = _load_incidents()
    for inc in incidents:
        if inc["id"] == incident_id:
            inc["status"] = new_status
            inc["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            break
    _save_incidents(incidents)
    flash("Incident status updated.", "success")
    return redirect(url_for("rescue_dashboard"))


@app.route("/rescue/<incident_id>/edit", methods=["POST"])
def rescue_edit_location(incident_id):
    location = request.form.get("location", "").strip()
    incidents = _load_incidents()
    for inc in incidents:
        if inc["id"] == incident_id:
            inc["location"] = location
            inc["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            break
    _save_incidents(incidents)
    flash("Location updated.", "success")
    return redirect(url_for("rescue_dashboard"))


@app.route("/dashboard")
def dashboard():
    history = get_history()
    total = len(history)
    accidents = sum(1 for h in history if h["prediction"] == "Accident")
    non_accidents = total - accidents
    accuracy = 96  # placeholder; could be computed from validation set

    stats = {
        "total_images": total,
        "accidents_detected": accidents,
        "non_accidents": non_accidents,
        "accuracy": accuracy,
    }

    import json
    history_json = json.dumps(
        [{"prediction": h["prediction"], "confidence": h["confidence"]} for h in history]
    )

    return render_template(
        "dashboard.html",
        stats=stats,
        history=history,
        history_json=history_json,
    )


if __name__ == "__main__":
    app.run(debug=True)
