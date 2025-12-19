# did_plus_mediapipe.py
import cv2
import numpy as np
import librosa
import easyocr
import json
from datetime import datetime
from typing import Dict, Any, Optional
import mediapipe as mp
import os

# ================== UTILS ==================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype("float64")
    b = b.flatten().astype("float64")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# ================== FACE MODULE (MediaPipe) ==================

class FaceModule:
    """
    Face embedding using MediaPipe FaceMesh.
    We extract 468 3D landmarks, flatten them, normalize, then apply
    a deterministic random projection to produce a 256-D embedding.
    """
    def __init__(self, embedding_dim: int = 256, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        # Deterministic random projection matrix
        rng = np.random.default_rng(seed)
        # projection: (468*3) -> embedding_dim
        self.landmark_size = 468 * 3
        self.proj = rng.standard_normal((self.landmark_size, embedding_dim)).astype("float32")
        # normalize columns for stable embeddings
        self.proj = self.proj / (np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-12)
        print(f"[Face] MediaPipe FaceMesh ready (embedding_dim={embedding_dim})")

    def _get_landmarks(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype="float32")  # shape (468,3)
        return pts  # (468,3)

    def detect_largest_face(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        We return the original image. The embedding uses landmarks on whole image.
        For systems that need cropping, you can implement cropping using bbox from landmarks.
        """
        return img_bgr

    def embed_face(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Returns embedding (embedding_dim,) as float32
        """
        pts = self._get_landmarks(img_bgr)
        if pts is None:
            raise RuntimeError("No face landmarks detected")
        flat = pts.flatten()  # length 468*3
        # normalize by centroid + scale
        flat_mean = flat.mean()
        flat = flat - flat_mean
        norm = np.linalg.norm(flat) + 1e-12
        flat = flat / norm
        # Pad/truncate to expected size (defensive)
        if flat.shape[0] < self.landmark_size:
            flat = np.pad(flat, (0, self.landmark_size - flat.shape[0]), mode="constant")
        elif flat.shape[0] > self.landmark_size:
            flat = flat[:self.landmark_size]
        # deterministic projection
        emb = np.dot(flat.astype("float32"), self.proj)  # (embedding_dim,)
        # final normalize
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb.astype("float32")

# ================== VOICE MODULE ==================

class VoiceModule:
    """Voice embedding using MFCC mean+std (80-D) — lightweight."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        print("[Voice] MFCC-based voice module ready")

    def embed_voice(self, audio_path: str) -> np.ndarray:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        if y.size == 0:
            raise ValueError("Empty audio file")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        emb = np.concatenate([mfcc_mean, mfcc_std]).astype("float32")  # 80-D
        # normalize
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

# ================== DOCUMENT MODULE ==================

class DocumentModule:
    """OCR (EasyOCR) + ID-card face embedding via MediaPipe landmarks"""
    def __init__(self, face_module: FaceModule, ocr_langs: list = ["en"]):
        print("[Doc] Initializing EasyOCR (may be slow first time)...")
        self.reader = easyocr.Reader(ocr_langs, gpu=False)
        self.face_module = face_module
        print("[Doc] Document module ready")

    def ocr_document(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Document image not found: {image_path}")
        result = self.reader.readtext(image_path, detail=1)
        full_text = " ".join([r[1] for r in result])
        return {"raw_ocr": result, "text": full_text}

    def extract_id_face_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempt to find face landmarks within the document image and return embedding.
        If none found, return None.
        """
        try:
            emb = self.face_module.embed_face(image_bgr)
            return emb
        except RuntimeError:
            return None

# ================== IDENTITY SYSTEM ==================

class IdentitySystem:
    def __init__(self):
        self.face_mod = FaceModule()
        self.voice_mod = VoiceModule()
        self.doc_mod = DocumentModule(self.face_mod)

    def enroll_user(self, user_id: str, face_path: str, voice_path: str, doc_path: str, template_file: str):
        print(f"[Enroll] {user_id} — face:{face_path} voice:{voice_path} doc:{doc_path}")
        # --- face
        img_face = cv2.imread(face_path)
        if img_face is None:
            raise FileNotFoundError(f"Face image not found: {face_path}")
        face_roi = self.face_mod.detect_largest_face(img_face)
        face_emb = self.face_mod.embed_face(face_roi)

        # --- voice
        voice_emb = self.voice_mod.embed_voice(voice_path)

        # --- document
        img_doc = cv2.imread(doc_path)
        if img_doc is None:
            raise FileNotFoundError(f"Document image not found: {doc_path}")
        doc_ocr = self.doc_mod.ocr_document(doc_path)
        id_face_emb = self.doc_mod.extract_id_face_embedding(img_doc)

        template = {
            "user_id": user_id,
            "enrolled_at": datetime.now().isoformat(),
            "face_embedding": face_emb.tolist(),
            "voice_embedding": voice_emb.tolist(),
            "id_face_embedding": None if id_face_emb is None else id_face_emb.tolist(),
            "document_text": doc_ocr["text"]
        }

        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)

        print("[Enroll] Template saved to", template_file)

    def verify_user(self, template_file: str, face_path: str, voice_path: str, doc_path: str) -> Dict[str, Any]:
        print("[Verify] Loading template:", template_file)
        with open(template_file, "r", encoding="utf-8") as f:
            tpl = json.load(f)

        stored_face = np.array(tpl["face_embedding"], dtype="float32")
        stored_voice = np.array(tpl["voice_embedding"], dtype="float32")
        stored_id_face = np.array(tpl["id_face_embedding"], dtype="float32") if tpl.get("id_face_embedding") else None
        stored_doc_text = tpl.get("document_text", "")

        # --- face
        img_face = cv2.imread(face_path)
        if img_face is None:
            raise FileNotFoundError(f"Face image not found: {face_path}")
        cur_face_emb = self.face_mod.embed_face(img_face)
        face_score = (cosine_similarity(stored_face, cur_face_emb) + 1.0) / 2.0

        # --- voice
        cur_voice_emb = self.voice_mod.embed_voice(voice_path)
        voice_score = (cosine_similarity(stored_voice, cur_voice_emb) + 1.0) / 2.0

        # --- document OCR text similarity (simple overlap)
        img_doc = cv2.imread(doc_path)
        if img_doc is None:
            raise FileNotFoundError(f"Document image not found: {doc_path}")
        cur_doc = self.doc_mod.ocr_document(doc_path)
        cur_text = cur_doc["text"]
        stored_words = set(stored_doc_text.lower().split())
        cur_words = set(cur_text.lower().split())
        doc_text_score = (len(stored_words & cur_words) / len(stored_words)) if stored_words else 0.0

        # --- id face on document
        id_face_score = 0.0
        if stored_id_face is not None:
            cur_id_emb = self.doc_mod.extract_id_face_embedding(img_doc)
            if cur_id_emb is not None:
                id_face_score = (cosine_similarity(stored_id_face, cur_id_emb) + 1.0) / 2.0

        doc_score = 0.7 * doc_text_score + 0.3 * id_face_score

        # --- fusion
        w_face, w_voice, w_doc = 0.4, 0.35, 0.25
        final_score = w_face * face_score + w_voice * voice_score + w_doc * doc_score

        result = {
            "face_score": float(face_score),
            "voice_score": float(voice_score),
            "doc_text_score": float(doc_text_score),
            "id_face_score": float(id_face_score),
            "doc_score": float(doc_score),
            "final_score": float(final_score),
            "verified": bool(final_score >= 0.75)
        }
        return result

# ================== CLI MAIN ==================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DID++ (MediaPipe) - enroll / verify demo")
    parser.add_argument("--mode", choices=["enroll", "verify"], required=True)
    parser.add_argument("--user_id", type=str, default="user_1")
    parser.add_argument("--template", type=str, default="template_user_1.json")
    parser.add_argument("--face", type=str, required=True, help="Path to face image (jpg/png)")
    parser.add_argument("--voice", type=str, required=True, help="Path to voice wav file")
    parser.add_argument("--doc", type=str, required=True, help="Path to document image (jpg/png)")
    args = parser.parse_args()

    system = IdentitySystem()

    if args.mode == "enroll":
        system.enroll_user(args.user_id, args.face, args.voice, args.doc, args.template)
    else:
        res = system.verify_user(args.template, args.face, args.voice, args.doc)
        print("\n[Verify] Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
