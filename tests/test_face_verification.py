import cv2
import datetime
import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from fpdf import FPDF
from sklearn.datasets import fetch_lfw_pairs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multimodal import FaceModule, cosine_similarity


# =========================
# Test Configuration
# =========================
NUM_TEST_SAMPLES = 20          # total samples (must be even)
THRESHOLD = 0.7
MAX_PDF_PER_CLASS = 5          # how many true + false to show in PDF


# =========================
# Helper Functions
# =========================

def save_pair_image(img1, img2, path):
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_score_histogram(results, path):
    scores = [r["score"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=25)
    plt.title("Similarity Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def confidence_score(r):
    """
    Higher = more confident example
    """
    if r["expected"]:          # same person
        return r["score"]
    else:                      # different people
        return 1.0 - r["score"]


def select_top_examples(results, max_per_class):
    true_samples = [r for r in results if r["expected"]]
    false_samples = [r for r in results if not r["expected"]]

    true_samples = sorted(true_samples, key=confidence_score, reverse=True)[:max_per_class]
    false_samples = sorted(false_samples, key=confidence_score, reverse=True)[:max_per_class]

    return true_samples + false_samples


# =========================
# PDF REPORT
# =========================

def generate_styled_pdf_report(
    detailed_results,
    X,
    total,
    valid,
    skipped,
    correct,
    accuracy
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # =========================
    # PAGE 1 — SUMMARY
    # =========================
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "LFW Face Verification Report", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now()}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, "Key Metrics", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Accuracy: {accuracy:.3f}", ln=True)
    pdf.cell(0, 8, f"Total Pairs: {total}", ln=True)
    pdf.cell(0, 8, f"Processed Pairs: {valid}", ln=True)
    pdf.cell(0, 8, f"Skipped Pairs: {skipped}", ln=True)
    pdf.ln(6)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        save_score_histogram(detailed_results, tmp.name)
        pdf.image(tmp.name, w=170)

    # =========================
    # PAGE 2 — TOP CONFIDENT EXAMPLES
    # =========================
    top_examples = select_top_examples(detailed_results, MAX_PDF_PER_CLASS)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "Top Confident True & False Examples", ln=True)
    pdf.ln(4)

    for r in top_examples:
        idx = r["pair_idx"]
        img1, img2 = X[idx]

        img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_pair_image(img1, img2, tmp.name)
            pdf.image(tmp.name, w=110)

        label = "SAME PERSON" if r["expected"] else "DIFFERENT PEOPLE"
        correctness = "Correct" if r["is_correct"] else "Wrong"

        pdf.set_font("Arial", size=10)
        pdf.multi_cell(
            0,
            6,
            f"Score: {r['score']:.3f} | Label: {label} | "
            f"Predicted: {r['predicted']} | {correctness}"
        )
        pdf.ln(4)

    filename = f"lfw_face_verification_report_{NUM_TEST_SAMPLES}.pdf"
    pdf.output(filename)
    print(f"PDF report generated: {filename}")


# =========================
# MAIN TEST
# =========================

def test_face_verification_lfw():
    """
    Face verification test on LFW dataset.
    Balanced sampling + dynamic PDF.
    """
    lfw_pairs = fetch_lfw_pairs(subset="test", color=True, resize=0.5)
    X, y = lfw_pairs.pairs, lfw_pairs.target

    # =========================
    # Balanced Subsampling
    # =========================
    if NUM_TEST_SAMPLES is not None:
        half = NUM_TEST_SAMPLES // 2

        true_idxs = [i for i, label in enumerate(y) if label == 1]
        false_idxs = [i for i, label in enumerate(y) if label == 0]

        np.random.shuffle(true_idxs)
        np.random.shuffle(false_idxs)

        selected_idxs = true_idxs[:half] + false_idxs[:half]
        np.random.shuffle(selected_idxs)

        X = X[selected_idxs]
        y = y[selected_idxs]

    face_mod = FaceModule()

    total = len(X)
    correct = 0
    skipped = 0
    detailed_results = []

    for idx, (img1, img2) in enumerate(X):
        img1_bgr = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        try:
            emb1 = face_mod.embed_face(img1_bgr)
            emb2 = face_mod.embed_face(img2_bgr)
        except RuntimeError:
            skipped += 1
            continue

        score = (cosine_similarity(emb1, emb2) + 1.0) / 2.0
        predicted = score >= THRESHOLD
        expected = bool(y[idx])
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        detailed_results.append({
            "pair_idx": idx,
            "score": score,
            "predicted": predicted,
            "expected": expected,
            "is_correct": is_correct
        })

    valid = total - skipped
    accuracy = correct / valid if valid > 0 else 0.0

    print("\n========== LFW FACE VERIFICATION REPORT ==========")
    print(f"Total pairs: {total}")
    print(f"Processed pairs: {valid}")
    print(f"Skipped pairs: {skipped}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.3f}")

    load_dotenv()
    if os.environ.get("LFW_PDF_REPORT", "1") == "1":
        generate_styled_pdf_report(
            detailed_results,
            X,
            total,
            valid,
            skipped,
            correct,
            accuracy
        )

    assert accuracy > 0.5, f"Accuracy too low: {accuracy:.3f}"
