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
import sys
import traceback

# Try to import soundfile, fallback to librosa
try:
    import soundfile as sf
    USE_SOUNDFILE = True
    print("[Audio] Using soundfile backend", flush=True)
except ImportError:
    USE_SOUNDFILE = False
    print("[Audio] Using librosa backend (install soundfile for better compatibility)", flush=True)

# ================== UTILS ==================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype("float64")
    b = b.flatten().astype("float64")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# ================== FACE MODULE ==================

class FaceModule:
    def __init__(self, embedding_dim: int = 256, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        rng = np.random.default_rng(seed)
        self.landmark_size = 468 * 3
        self.proj = rng.standard_normal((self.landmark_size, embedding_dim)).astype("float32")
        self.proj /= (np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-12)
        print(f"[Face] Ready (embedding_dim={embedding_dim})", flush=True)

    def _get_landmarks(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            if image_bgr is None or image_bgr.size == 0:
                print("[ERROR] Invalid image input", flush=True)
                return None
            
            # Resize if too large
            h, w = image_bgr.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image_bgr = cv2.resize(image_bgr, (new_w, new_h))
                print(f"[Face] Resized to {new_w}x{new_h}", flush=True)
            
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = self.mp_face.process(rgb)
            
            if not res.multi_face_landmarks:
                print("[WARN] No face landmarks detected", flush=True)
                return None
            
            lm = res.multi_face_landmarks[0].landmark
            landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype="float32")
            print(f"[Face] Detected {len(landmarks)} landmarks", flush=True)
            return landmarks
            
        except Exception as e:
            print(f"[ERROR] Landmark extraction failed: {e}", flush=True)
            traceback.print_exc()
            return None

    def embed_face(self, img_bgr: np.ndarray) -> np.ndarray:
        pts = self._get_landmarks(img_bgr)
        if pts is None:
            raise RuntimeError("No face detected")

        flat = pts.flatten()
        flat -= flat.mean()
        flat /= (np.linalg.norm(flat) + 1e-12)

        emb = np.dot(flat[: self.landmark_size], self.proj)
        emb /= (np.linalg.norm(emb) + 1e-12)
        return emb.astype("float32")

# ================== VOICE MODULE ==================

class VoiceModule:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        backend = "soundfile" if USE_SOUNDFILE else "librosa"
        print(f"[Voice] MFCC ready (backend: {backend})", flush=True)

    def embed_voice(self, audio_path: str) -> np.ndarray:
        try:
            # Check file exists
            if not os.path.exists(audio_path):
                abs_path = os.path.abspath(audio_path)
                print(f"[ERROR] Audio file not found: {abs_path}", flush=True)
                print(f"[HINT] Current directory: {os.getcwd()}", flush=True)
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise RuntimeError(f"Audio file is empty (0 bytes): {audio_path}")
            
            print(f"[Voice] Loading: {audio_path} ({file_size} bytes)", flush=True)

            # Load audio with appropriate backend
            if USE_SOUNDFILE:
                # Use soundfile directly
                y, sr = sf.read(audio_path, always_2d=False)
                print(f"[Voice] soundfile load: shape={y.shape}, sr={sr}, dtype={y.dtype}", flush=True)

                # Handle stereo
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                    print(f"[Voice] Converted stereo to mono", flush=True)

                # Convert to float32
                if y.dtype != np.float32:
                    y = y.astype(np.float32)

                # Resample if needed
                if sr != self.sample_rate:
                    print(f"[Voice] Resampling {sr}Hz → {self.sample_rate}Hz", flush=True)
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
            else:
                # Use librosa
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                print(f"[Voice] librosa load: {len(y)} samples, {len(y)/sr:.2f}s", flush=True)

            # Validate audio
            if y.size == 0:
                raise RuntimeError("Audio has no samples after loading")
            
            if len(y) < 1000:
                print(f"[WARN] Audio is very short: {len(y)} samples ({len(y)/sr:.3f}s)", flush=True)

            # Extract MFCC features
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                print(f"[Voice] MFCC shape: {mfcc.shape}", flush=True)
            except Exception as mfcc_error:
                print(f"[ERROR] MFCC extraction failed: {mfcc_error}", flush=True)
                raise RuntimeError(f"Failed to extract MFCC: {mfcc_error}")

            # Create embedding
            emb = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            emb /= (np.linalg.norm(emb) + 1e-12)

            print(f"[Voice] ✓ Embedding created: {emb.shape}", flush=True)
            return emb.astype("float32")

        except FileNotFoundError:
            raise
        except Exception as e:
            print(f"[FATAL] Voice embedding failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise

# ================== DOCUMENT MODULE ==================

class DocumentModule:
    def __init__(self, face_module: FaceModule, ocr_langs=["en"]):
        print("[Doc] Initializing EasyOCR (safe mode)", flush=True)
        try:
            self.reader = easyocr.Reader(ocr_langs, gpu=False, verbose=False)
            self.ocr_enabled = True
            print("[Doc] ✓ OCR ready", flush=True)
        except Exception as e:
            print(f"[WARN] EasyOCR initialization failed: {e}", flush=True)
            print(f"[WARN] OCR will be disabled", flush=True)
            self.reader = None
            self.ocr_enabled = False
        self.face_module = face_module

    def ocr_document(self, image_path: str) -> Dict[str, Any]:
        if not self.ocr_enabled:
            return {"text": ""}

        try:
            if not os.path.exists(image_path):
                print(f"[WARN] Document not found: {image_path}", flush=True)
                return {"text": ""}
            
            result = self.reader.readtext(image_path, detail=1)
            text = " ".join([r[1] for r in result])
            print(f"[Doc] OCR extracted {len(text)} characters", flush=True)
            return {"text": text}
        except Exception as e:
            print(f"[WARN] OCR failed: {e}", flush=True)
            return {"text": ""}

    def extract_id_face_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            print("[Doc] Extracting face from ID document...", flush=True)
            emb = self.face_module.embed_face(image_bgr)
            print(f"[Doc] ✓ ID face embedding: {emb.shape}", flush=True)
            return emb
        except Exception as e:
            print(f"[WARN] Could not extract face from ID: {e}", flush=True)
            return None

# ================== IDENTITY SYSTEM ==================

class IdentitySystem:
    def __init__(self):
        print("\n" + "="*60)
        print("DID++ Identity System - Initializing")
        print("="*60)
        self.face_mod = FaceModule()
        self.voice_mod = VoiceModule()
        self.doc_mod = DocumentModule(self.face_mod)
        print("="*60)
        print("✓ System initialized successfully!")
        print("="*60 + "\n")

    def enroll_user(self, user_id, face_path, voice_path, doc_path, template_file):
        print(f"\n{'='*60}")
        print(f"ENROLLMENT: {user_id}")
        print(f"{'='*60}")

        # Convert to absolute paths
        face_path = os.path.abspath(face_path)
        voice_path = os.path.abspath(voice_path)
        doc_path = os.path.abspath(doc_path)
        template_file = os.path.abspath(template_file)

        # Validate files
        print("\n[1/4] Validating input files...")
        for path, name in [(face_path, "Face"), (voice_path, "Voice"), (doc_path, "Document")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
            print(f"  ✓ {name}: {os.path.basename(path)}")

        # Process face
        print("\n[2/4] Processing face image...")
        img_face = cv2.imread(face_path)
        if img_face is None:
            raise RuntimeError(f"Could not read face image: {face_path}")
        print(f"  Image: {img_face.shape[1]}x{img_face.shape[0]}")
        face_emb = self.face_mod.embed_face(img_face)
        print(f"  ✓ Face embedding: {face_emb.shape}")

        # Process voice
        print("\n[3/4] Processing voice audio...")
        voice_emb = self.voice_mod.embed_voice(voice_path)
        print(f"  ✓ Voice embedding: {voice_emb.shape}")

        # Process document
        print("\n[4/4] Processing ID document...")
        img_doc = cv2.imread(doc_path)
        if img_doc is None:
            print(f"  [WARN] Could not read document image")
            doc_ocr = {"text": ""}
            id_face_emb = None
        else:
            doc_ocr = self.doc_mod.ocr_document(doc_path)
            id_face_emb = self.doc_mod.extract_id_face_embedding(img_doc)

        # Create template
        template = {
            "user_id": user_id,
            "enrolled_at": datetime.now().isoformat(),
            "face_embedding": face_emb.tolist(),
            "voice_embedding": voice_emb.tolist(),
            "id_face_embedding": None if id_face_emb is None else id_face_emb.tolist(),
            "document_text": doc_ocr["text"]
        }

        # Save
        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ ENROLLMENT SUCCESSFUL")
        print(f"  Template: {os.path.basename(template_file)}")
        print(f"  Size: {os.path.getsize(template_file)} bytes")
        print(f"{'='*60}\n")

    def verify_user(self, template_file, face_path, voice_path, doc_path):
        print(f"\n{'='*60}")
        print("VERIFICATION IN PROGRESS")
        print(f"{'='*60}")

        # Load template
        print("\n[1/3] Loading template...")
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"Template not found: {template_file}")
        
        with open(template_file, "r", encoding="utf-8") as f:
            tpl = json.load(f)
        print(f"  ✓ Template loaded: {tpl['user_id']}")

        face_ref = np.array(tpl["face_embedding"])
        voice_ref = np.array(tpl["voice_embedding"])

        # Verify face
        print("\n[2/3] Verifying face...")
        img_face = cv2.imread(face_path)
        if img_face is None:
            raise RuntimeError(f"Could not read face image: {face_path}")
        face_emb = self.face_mod.embed_face(img_face)
        face_score = (cosine_similarity(face_ref, face_emb) + 1) / 2
        print(f"  Face score: {face_score:.2%}")

        # Verify voice
        print("\n[3/3] Verifying voice...")
        voice_emb = self.voice_mod.embed_voice(voice_path)
        voice_score = (cosine_similarity(voice_ref, voice_emb) + 1) / 2
        print(f"  Voice score: {voice_score:.2%}")

        # Final decision
        final_score = 0.6 * face_score + 0.4 * voice_score
        verified = final_score >= 0.75

        print(f"\n{'='*60}")
        if verified:
            print(f"✓ VERIFICATION SUCCESSFUL")
        else:
            print(f"✗ VERIFICATION FAILED")
        print(f"  Final score: {final_score:.2%} (threshold: 75%)")
        print(f"{'='*60}\n")

        return {
            "user_id": tpl["user_id"],
            "face_score": float(face_score),
            "voice_score": float(voice_score),
            "final_score": float(final_score),
            "verified": verified,
            "threshold": 0.75
        }

# ================== CLI ==================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DID++ Identity Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll
  python multimodal.py --mode enroll --user_id john \\
    --face face.jpg --voice voice.wav --doc id.jpg

  # Verify
  python multimodal.py --mode verify \\
    --face verify_face.jpg --voice verify_voice.wav --doc id.jpg \\
    --template template_user_1.json
        """
    )
    parser.add_argument("--mode", choices=["enroll", "verify"], required=True)
    parser.add_argument("--user_id", default="user_1")
    parser.add_argument("--template", default="template_user_1.json")
    parser.add_argument("--face", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--doc", required=True)
    args = parser.parse_args()

    system = IdentitySystem()

    if args.mode == "enroll":
        system.enroll_user(args.user_id, args.face, args.voice, args.doc, args.template)
    else:
        res = system.verify_user(args.template, args.face, args.voice, args.doc)
        print("\n[Verification Details]")
        for k, v in res.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Cancelled by user")
        sys.exit(130)
    except Exception:
        print("\n[FATAL ERROR]")
        traceback.print_exc()
        sys.exit(1)