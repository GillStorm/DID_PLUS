import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from multimodal import FaceModule

# ================= CONFIG =================

UPLOAD_DIR = "uploads"
MODEL_VERSION = "face-model@1.2.3"
POLICY_VERSION = "verifier-policy@2025-01"
VERIFY_THRESHOLD = 0.75

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= APP =================

app = Flask(__name__)
face_model = FaceModule()

# ================= UTILS =================

def cosine_sim(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

def confidence_bucket(score):
    if score >= 0.85:
        return "VERY_HIGH"
    elif score >= 0.75:
        return "HIGH"
    elif score >= 0.6:
        return "MEDIUM"
    return "LOW"

def gen_schema_id():
    return "0x" + uuid.uuid4().hex[:16]

def save_temp(file):
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    file.save(path)
    return path

# ================= UI ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/face-verification")
def face_verification():
    return render_template("face_verify.html")

# ================= UI VERIFY (STATELESS) =================

@app.route("/verify", methods=["POST"])
def verify():
    if "reference" not in request.files or "live" not in request.files:
        return jsonify({"error": "reference and live images required"}), 400

    ref_path = save_temp(request.files["reference"])
    live_path = save_temp(request.files["live"])

    try:
        ref_emb = face_model.embed_face(cv2.imread(ref_path))
        live_emb = face_model.embed_face(cv2.imread(live_path))

        score = (cosine_sim(ref_emb, live_emb) + 1.0) / 2.0

        return jsonify({
            "verified": score >= VERIFY_THRESHOLD,
            "final_score": score,
            "confidence_bucket": confidence_bucket(score),
            "model_version": MODEL_VERSION
        })

    finally:
        os.remove(ref_path)
        os.remove(live_path)

# ================= API: FULL TECHNICAL JSON =================

@app.route("/api/verify-face", methods=["POST"])
def api_verify_face():
    if "reference" not in request.files or "live" not in request.files:
        return jsonify({"error": "reference and live images required"}), 400

    ref_path = save_temp(request.files["reference"])
    live_path = save_temp(request.files["live"])

    try:
        ref_emb = face_model.embed_face(cv2.imread(ref_path))
        live_emb = face_model.embed_face(cv2.imread(live_path))

        score = (cosine_sim(ref_emb, live_emb) + 1.0) / 2.0

        return jsonify({
            "verified": score >= VERIFY_THRESHOLD,
            "confidence_bucket": confidence_bucket(score),
            "model_version": MODEL_VERSION,
            "scores": {
                "face_similarity": float(score)
            },
            "embeddings": {
                "reference_embedding": ref_emb.tolist(),
                "live_embedding": live_emb.tolist(),
                "embedding_dim": len(ref_emb)
            },
            "thresholds": {
                "verify_threshold": VERIFY_THRESHOLD
            }
        })

    finally:
        os.remove(ref_path)
        os.remove(live_path)

# ================= API: POLICY / SCHEMA =================

@app.route("/api/schema-verify", methods=["POST"])
def api_schema_verify():
    if "reference" not in request.files or "live" not in request.files:
        return jsonify({"error": "reference and live images required"}), 400

    ref_path = save_temp(request.files["reference"])
    live_path = save_temp(request.files["live"])

    try:
        ref_emb = face_model.embed_face(cv2.imread(ref_path))
        live_emb = face_model.embed_face(cv2.imread(live_path))

        score = (cosine_sim(ref_emb, live_emb) + 1.0) / 2.0

        return jsonify({
            "schema_id": gen_schema_id(),
            "verified": score >= VERIFY_THRESHOLD,
            "confidence_bucket": confidence_bucket(score),
            "model_version": MODEL_VERSION,
            "policy_version": POLICY_VERSION
        })

    finally:
        os.remove(ref_path)
        os.remove(live_path)

# ================= RUN =================

if __name__ == "__main__":
    print("[INFO] Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
