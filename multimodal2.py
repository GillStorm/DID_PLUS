# did_plus_mediapipe.py
import cv2
import numpy as np
import librosa
import easyocr
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import mediapipe as mp
import os
import sys
import traceback
import re

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

# ================== VOICE MODULE (ENHANCED) ==================

class VoiceModule:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        backend = "soundfile" if USE_SOUNDFILE else "librosa"
        print(f"[Voice] Enhanced speaker verification ready (backend: {backend})", flush=True)

    def extract_speaker_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive speaker-specific features
        Returns multiple feature types for robust speaker verification
        """
        features = {}
        
        # 1. MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features['mfcc_mean'] = mfcc.mean(axis=1)
        features['mfcc_std'] = mfcc.std(axis=1)
        features['mfcc_delta'] = librosa.feature.delta(mfcc).mean(axis=1)
        
        # 2. Pitch/F0 (Fundamental frequency - voice uniqueness)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.array([np.mean(pitch_values)])
            features['pitch_std'] = np.array([np.std(pitch_values)])
            features['pitch_range'] = np.array([np.max(pitch_values) - np.min(pitch_values)])
        else:
            features['pitch_mean'] = np.array([0.0])
            features['pitch_std'] = np.array([0.0])
            features['pitch_range'] = np.array([0.0])
        
        # 3. Spectral features (voice timbre)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        features['spectral_centroid_mean'] = np.array([np.mean(spectral_centroids)])
        features['spectral_centroid_std'] = np.array([np.std(spectral_centroids)])
        features['spectral_rolloff_mean'] = np.array([np.mean(spectral_rolloff)])
        features['spectral_bandwidth_mean'] = np.array([np.mean(spectral_bandwidth)])
        
        # 4. Zero Crossing Rate (voice quality)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.array([np.mean(zcr)])
        features['zcr_std'] = np.array([np.std(zcr)])
        
        # 5. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = chroma.mean(axis=1)
        
        # 6. Mel Spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spec_mean'] = mel_spec_db.mean(axis=1)
        features['mel_spec_std'] = mel_spec_db.std(axis=1)
        
        return features

    def embed_voice(self, audio_path: str) -> np.ndarray:
        try:
            if not os.path.exists(audio_path):
                abs_path = os.path.abspath(audio_path)
                print(f"[ERROR] Audio file not found: {abs_path}", flush=True)
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise RuntimeError(f"Audio file is empty (0 bytes): {audio_path}")
            
            print(f"[Voice] Loading: {audio_path} ({file_size} bytes)", flush=True)

            if USE_SOUNDFILE:
                y, sr = sf.read(audio_path, always_2d=False)
                print(f"[Voice] soundfile load: shape={y.shape}, sr={sr}", flush=True)

                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                    print(f"[Voice] Converted stereo to mono", flush=True)

                if y.dtype != np.float32:
                    y = y.astype(np.float32)

                if sr != self.sample_rate:
                    print(f"[Voice] Resampling {sr}Hz → {self.sample_rate}Hz", flush=True)
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
            else:
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                print(f"[Voice] librosa load: {len(y)} samples, {len(y)/sr:.2f}s", flush=True)

            if y.size == 0:
                raise RuntimeError("Audio has no samples after loading")
            
            duration = len(y) / sr
            if duration < 1.0:
                print(f"[WARN] Audio is very short: {duration:.2f}s - may affect accuracy", flush=True)
            elif duration < 0.5:
                raise RuntimeError(f"Audio too short for speaker verification: {duration:.2f}s (need at least 0.5s)")

            # Extract comprehensive speaker features
            print(f"[Voice] Extracting speaker-specific features...", flush=True)
            features = self.extract_speaker_features(y, sr)
            
            # Concatenate all features into a single embedding
            embedding_parts = []
            for key in sorted(features.keys()):
                embedding_parts.append(features[key])
            
            emb = np.concatenate(embedding_parts)
            
            # Normalize
            emb = emb - np.mean(emb)
            emb /= (np.linalg.norm(emb) + 1e-12)

            print(f"[Voice] ✓ Speaker embedding created: {emb.shape}", flush=True)
            print(f"[Voice] Features extracted: {len(features)} types", flush=True)
            
            return emb.astype("float32")

        except FileNotFoundError:
            raise
        except Exception as e:
            print(f"[FATAL] Voice embedding failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise

# ================== DOCUMENT MODULE (ENHANCED) ==================

class DocumentModule:
    def __init__(self, face_module: FaceModule, ocr_langs=["en"]):
        print("[Doc] Initializing document verification module...", flush=True)
        
        # Initialize OCR
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
        
        # Common ID document patterns
        self.patterns = {
            'name': [
                r'Name[:\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
                r'Full Name[:\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
                r'([A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
            ],
            'id_number': [
                r'ID[:\s#]+([A-Z0-9\-]+)',
                r'Number[:\s#]+([A-Z0-9\-]+)',
                r'Card No[:\s#]+([A-Z0-9\-]+)',
                r'\b([A-Z]{2}[0-9]{6,})\b',
            ],
            'dob': [
                r'DOB[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'Date of Birth[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'Born[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'expiry': [
                r'Exp[a-z]*[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'Valid Until[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'address': [
                r'Address[:\s]+([A-Za-z0-9\s,.-]+(?:\d{5,6})?)',
            ]
        }

    def ocr_document(self, image_path: str) -> Dict[str, Any]:
        """Extract text from document using OCR"""
        if not self.ocr_enabled:
            return {"text": "", "structured_data": {}}

        try:
            if not os.path.exists(image_path):
                print(f"[WARN] Document not found: {image_path}", flush=True)
                return {"text": "", "structured_data": {}}
            
            print(f"[Doc] Running OCR on: {image_path}", flush=True)
            result = self.reader.readtext(image_path, detail=1)
            
            # Extract all text
            full_text = " ".join([r[1] for r in result])
            print(f"[Doc] OCR extracted {len(full_text)} characters", flush=True)
            
            # Extract structured data
            structured_data = self.extract_structured_data(full_text)
            
            # Get text positions for quality check
            text_boxes = [(r[0], r[1], r[2]) for r in result]  # bbox, text, confidence
            avg_confidence = np.mean([r[2] for r in result]) if result else 0.0
            
            return {
                "text": full_text,
                "structured_data": structured_data,
                "text_boxes": text_boxes,
                "avg_confidence": float(avg_confidence),
                "num_fields": len(result)
            }
        except Exception as e:
            print(f"[WARN] OCR failed: {e}", flush=True)
            return {"text": "", "structured_data": {}}

    def extract_structured_data(self, text: str) -> Dict[str, str]:
        """Extract structured information from OCR text"""
        data = {}
        
        for field, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data[field] = match.group(1).strip()
                    break
        
        return data

    def extract_id_face_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract face from ID document photo"""
        try:
            print("[Doc] Extracting face from ID document...", flush=True)
            emb = self.face_module.embed_face(image_bgr)
            print(f"[Doc] ✓ ID face embedding: {emb.shape}", flush=True)
            return emb
        except Exception as e:
            print(f"[WARN] Could not extract face from ID: {e}", flush=True)
            return None

    def check_document_quality(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Check document image quality"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            # Check sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Check brightness
            brightness = np.mean(gray)
            
            # Check contrast
            contrast = gray.std()
            
            # Detect edges (for document boundaries)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            quality_score = 0.0
            issues = []
            
            # Sharpness check
            if sharpness < 100:
                issues.append("Image is blurry")
            else:
                quality_score += 0.3
            
            # Brightness check
            if brightness < 50:
                issues.append("Image is too dark")
            elif brightness > 200:
                issues.append("Image is too bright")
            else:
                quality_score += 0.3
            
            # Contrast check
            if contrast < 30:
                issues.append("Low contrast")
            else:
                quality_score += 0.2
            
            # Edge density check
            if edge_density > 0.1:
                quality_score += 0.2
            
            return {
                'quality_score': float(quality_score),
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'edge_density': float(edge_density),
                'is_acceptable': quality_score >= 0.6,
                'issues': issues
            }
            
        except Exception as e:
            print(f"[WARN] Quality check failed: {e}", flush=True)
            return {'quality_score': 0.0, 'is_acceptable': False, 'issues': ['Quality check failed']}

    def verify_document(self, enrolled_doc_data: Dict, enrolled_live_face: np.ndarray,
                       current_image_path: str, current_image_bgr: np.ndarray,
                       current_live_face: np.ndarray) -> Dict[str, Any]:
        """
        Verify document by comparing:
        1. Face on ID document with live face photo
        2. Extracted text consistency
        """
        results = {
            'id_to_live_match': False,
            'id_to_live_score': 0.0,
            'text_match': False,
            'text_similarity': 0.0,
            'document_score': 0.0,
            'verified': False
        }
        
        try:
            print(f"[Doc] Verifying document...", flush=True)
            
            # 1. Extract face from current ID document
            print(f"[Doc] Extracting face from ID document...", flush=True)
            current_id_face = self.extract_id_face_embedding(current_image_bgr)
            
            # 2. Compare ID face with live photo face
            if current_id_face is not None:
                print(f"[Doc] Comparing ID photo with live face photo...", flush=True)
                
                # Compare current ID face with current live face
                face_similarity = cosine_similarity(current_id_face, current_live_face)
                id_to_live_score = (face_similarity + 1) / 2
                
                results['id_to_live_score'] = float(id_to_live_score)
                results['id_to_live_match'] = id_to_live_score >= 0.75  # 75% threshold
                
                print(f"[Doc] ID-to-Live face match: {id_to_live_score:.2%}", flush=True)
            else:
                print(f"[WARN] Could not extract face from ID document", flush=True)
            
            # 3. Extract and compare text (optional, for additional verification)
            print(f"[Doc] Comparing document text...", flush=True)
            current_ocr = self.ocr_document(current_image_path)
            enrolled_text = enrolled_doc_data.get('document_text', '')
            current_text = current_ocr.get('text', '')
            
            if enrolled_text and current_text:
                # Simple text similarity (Jaccard similarity)
                enrolled_words = set(enrolled_text.lower().split())
                current_words = set(current_text.lower().split())
                
                if enrolled_words and current_words:
                    intersection = enrolled_words & current_words
                    union = enrolled_words | current_words
                    text_similarity = len(intersection) / len(union)
                    
                    results['text_similarity'] = float(text_similarity)
                    results['text_match'] = text_similarity >= 0.70  # 70% threshold
                    
                    print(f"[Doc] Text similarity: {text_similarity:.2%}", flush=True)
                    
                    # Compare structured data if available
                    if current_ocr.get('structured_data'):
                        results['structured_data'] = current_ocr['structured_data']
            
            # 4. Calculate document score (weighted average)
            # 70% ID-to-Live face match + 30% text similarity
            document_score = (
                0.70 * results['id_to_live_score'] +
                0.30 * results.get('text_similarity', 0.0)
            )
            results['document_score'] = float(document_score)
            
            # Document is verified if:
            # - ID face matches live face (>= 75%) OR
            # - Text matches (>= 70%)
            results['verified'] = (
                results['id_to_live_match'] or results['text_match']
            )
            
            print(f"[Doc] Document score: {document_score:.2%}", flush=True)
            print(f"[Doc] Document verification: {'PASS' if results['verified'] else 'FAIL'}", flush=True)
            
        except Exception as e:
            print(f"[ERROR] Document verification failed: {e}", flush=True)
            traceback.print_exc()
        
        return results

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

        face_path = os.path.abspath(face_path)
        voice_path = os.path.abspath(voice_path)
        doc_path = os.path.abspath(doc_path)
        template_file = os.path.abspath(template_file)

        print("\n[1/4] Validating input files...")
        for path, name in [(face_path, "Face"), (voice_path, "Voice"), (doc_path, "Document")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
            print(f"  ✓ {name}: {os.path.basename(path)}")

        print("\n[2/4] Processing face image...")
        img_face = cv2.imread(face_path)
        if img_face is None:
            raise RuntimeError(f"Could not read face image: {face_path}")
        print(f"  Image: {img_face.shape[1]}x{img_face.shape[0]}")
        face_emb = self.face_mod.embed_face(img_face)
        print(f"  ✓ Face embedding: {face_emb.shape}")

        print("\n[3/4] Processing voice audio...")
        voice_emb = self.voice_mod.embed_voice(voice_path)
        print(f"  ✓ Voice embedding: {voice_emb.shape}")

        print("\n[4/4] Processing ID document...")
        img_doc = cv2.imread(doc_path)
        if img_doc is None:
            print(f"  [WARN] Could not read document image")
            doc_ocr = {"text": "", "structured_data": {}}
            id_face_emb = None
        else:
            # Extract text and structured data
            doc_ocr = self.doc_mod.ocr_document(doc_path)
            print(f"  OCR confidence: {doc_ocr.get('avg_confidence', 0):.2%}")
            
            # Show extracted structured data
            if doc_ocr.get('structured_data'):
                print(f"  Structured data extracted:")
                for key, value in doc_ocr['structured_data'].items():
                    print(f"    • {key}: {value}")
            
            # Extract face from ID
            id_face_emb = self.doc_mod.extract_id_face_embedding(img_doc)

        template = {
            "user_id": user_id,
            "enrolled_at": datetime.now().isoformat(),
            "face_embedding": face_emb.tolist(),
            "voice_embedding": voice_emb.tolist(),
            "id_face_embedding": None if id_face_emb is None else id_face_emb.tolist(),
            "document_text": doc_ocr.get("text", ""),
            "document_structured": doc_ocr.get("structured_data", {})
        }

        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ ENROLLMENT SUCCESSFUL")
        print(f"  Template: {os.path.basename(template_file)}")
        print(f"  Size: {os.path.getsize(template_file):,} bytes")
        print(f"{'='*60}\n")

    def verify_user(self, template_file, face_path, voice_path, doc_path):
        print(f"\n{'='*60}")
        print("VERIFICATION IN PROGRESS")
        print(f"{'='*60}")

        print("\n[1/4] Loading template...")
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"Template not found: {template_file}")
        
        with open(template_file, "r", encoding="utf-8") as f:
            tpl = json.load(f)
        print(f"  ✓ Template loaded: {tpl['user_id']}")

        face_ref = np.array(tpl["face_embedding"])
        voice_ref = np.array(tpl["voice_embedding"])

        print("\n[2/4] Verifying face...")
        img_face = cv2.imread(face_path)
        if img_face is None:
            raise RuntimeError(f"Could not read face image: {face_path}")
        face_emb = self.face_mod.embed_face(img_face)
        face_score = (cosine_similarity(face_ref, face_emb) + 1) / 2
        print(f"  Face score: {face_score:.2%}")

        print("\n[3/4] Verifying voice...")
        voice_emb = self.voice_mod.embed_voice(voice_path)
        
        # Use Euclidean distance for voice (more discriminative than cosine for speaker verification)
        euclidean_dist = np.linalg.norm(voice_ref - voice_emb)
        
        # Convert distance to similarity score (0-1 range)
        # Typical different speaker distance: 0.4-0.8
        # Typical same speaker distance: 0.1-0.3
        # Using exponential decay: score = e^(-k*distance)
        voice_score = np.exp(-3.0 * euclidean_dist)
        
        # Also compute cosine similarity for comparison
        cosine_sim = (cosine_similarity(voice_ref, voice_emb) + 1) / 2
        
        # Use weighted combination (favor distance-based metric)
        voice_score = 0.7 * voice_score + 0.3 * cosine_sim
        
        print(f"  Voice score: {voice_score:.2%} (euclidean: {euclidean_dist:.3f}, cosine: {cosine_sim:.2%})")
        print(f"  Speaker match: {'SAME' if voice_score >= 0.70 else 'DIFFERENT'}")

        print("\n[4/4] Verifying document...")
        img_doc = cv2.imread(doc_path)
        if img_doc is None:
            print(f"  [WARN] Could not read document image")
            doc_results = {
                'verified': False,
                'document_score': 0.0,
                'id_to_live_score': 0.0,
                'text_similarity': 0.0
            }
        else:
            doc_results = self.doc_mod.verify_document(tpl, face_emb, doc_path, img_doc, face_emb)
        
        print(f"  Document score: {doc_results['document_score']:.2%}")
        print(f"  ID-to-Live face match: {doc_results['id_to_live_score']:.2%}")
        if doc_results.get('text_similarity', 0) > 0:
            print(f"  Text similarity: {doc_results['text_similarity']:.2%}")

        # Multi-modal scoring: 45% Face + 35% Voice + 20% Document
        final_score = (
            0.45 * face_score +
            0.35 * voice_score +
            0.20 * doc_results['document_score']
        )
        
        verified = (
            face_score >= 0.75 and
            voice_score >= 0.70 and
            doc_results['document_score'] >= 0.70 and
            final_score >= 0.75
        )

        print(f"\n{'='*60}")
        if verified:
            print(f"✓ VERIFICATION SUCCESSFUL")
        else:
            print(f"✗ VERIFICATION FAILED")
        print(f"  Final score: {final_score:.2%}")
        print(f"  Weights: Face 45% + Voice 35% + Document 20%")
        print(f"  Thresholds: Face≥75%, Voice≥70%, Document≥70%, Overall≥75%")
        print(f"{'='*60}\n")

        return {
            "user_id": tpl["user_id"],
            "face_score": float(face_score),
            "voice_score": float(voice_score),
            "document_score": doc_results['document_score'],
            "id_to_live_face_score": doc_results['id_to_live_score'],
            "document_text_similarity": doc_results.get('text_similarity', 0.0),
            "final_score": float(final_score),
            "verified": verified,
            "weights": {
                "face": 0.45,
                "voice": 0.35,
                "document": 0.20
            },
            "checks_passed": {
                "face": face_score >= 0.75,
                "voice": voice_score >= 0.70,
                "document": doc_results['document_score'] >= 0.70,
                "overall": final_score >= 0.75
            }
        }

# ================== CLI ==================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DID++ Identity Verification System with Document Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll
  python multimodal.py --mode enroll --user_id john \\
    --face face.jpg --voice voice.wav --doc id_card.jpg

  # Verify
  python multimodal.py --mode verify \\
    --face verify_face.jpg --voice verify_voice.wav --doc verify_id.jpg \\
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
