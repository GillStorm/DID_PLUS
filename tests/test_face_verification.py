
import os
import sys
import json
import cv2
import numpy as np
import pytest
from fpdf import FPDF
from sklearn.datasets import fetch_lfw_pairs
from collections import Counter
import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import FaceModule, cosine_similarity

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')
PAIRS_FILE = os.path.join(SAMPLE_DATA_DIR, 'pairs.json')


def test_face_verification_lfw():
    """
    Test face verification on a large number of pairs from the LFW dataset.
    Set the environment variable LFW_PDF_REPORT=1 to generate a PDF report.
    """
    lfw_pairs = fetch_lfw_pairs(subset='test', color=True, resize=0.5)
    X, y = lfw_pairs.pairs, lfw_pairs.target
    face_mod = FaceModule()
    correct = 0
    total = len(X)
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
        result = score >= 0.7
        expected = bool(y[idx])
        is_correct = result == expected
        if is_correct:
            correct += 1
        detailed_results.append({
            'pair_idx': idx,
            'score': score,
            'predicted': result,
            'expected': expected,
            'is_correct': is_correct
        })
    valid = total - skipped
    accuracy = correct / valid if valid > 0 else 0
    # Verbose report
    report_lines = []
    report_lines.append("\n================ LFW Face Verification Detailed Report ================")
    report_lines.append(f"Test Date: {datetime.datetime.now()}")
    report_lines.append(f"Total pairs: {total}")
    report_lines.append(f"Pairs processed: {valid}")
    report_lines.append(f"Pairs skipped: {skipped}")
    report_lines.append(f"Correct predictions: {correct}")
    report_lines.append(f"Accuracy: {accuracy:.3f}")
    # Statistics
    pred_counter = Counter([r['predicted'] for r in detailed_results])
    exp_counter = Counter([r['expected'] for r in detailed_results])
    report_lines.append(f"\nPrediction breakdown: {pred_counter}")
    report_lines.append(f"Expected breakdown: {exp_counter}")
    report_lines.append("====================================================================\n")
    # Print the report to stdout (so pytest will show it even on pass)
    print("\n".join(report_lines))
    generate_pdf_report = os.environ.get("LFW_PDF_REPORT", "0") == "1"
    if generate_pdf_report:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="LFW Face Verification Test Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Test Date: {datetime.datetime.now()}", ln=True)
            pdf.cell(200, 10, txt=f"Total pairs: {total}", ln=True)
            pdf.cell(200, 10, txt=f"Pairs processed: {valid}", ln=True)
            pdf.cell(200, 10, txt=f"Pairs skipped: {skipped}", ln=True)
            pdf.cell(200, 10, txt=f"Correct predictions: {correct}", ln=True)
            pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.3f}", ln=True)
            pdf.cell(200, 10, txt=f"Prediction breakdown: {dict(pred_counter)}", ln=True)
            pdf.cell(200, 10, txt=f"Expected breakdown: {dict(exp_counter)}", ln=True)
            pdf.output("lfw_face_verification_report.pdf")
            print("PDF report generated: lfw_face_verification_report.pdf")
        except ImportError:
            print("[ERROR] fpdf not installed. Run 'pip install fpdf' to enable PDF report generation.")
    assert accuracy > 0.5, f"Accuracy too low: {accuracy:.3f} (Skipped {skipped} of {total})"
