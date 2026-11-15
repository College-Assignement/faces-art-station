"""
Comprehensive Test Suite for Face Shape Refinement in F.A.C.E.S.
Verifies all 7 Face Shapes:
1. Oval
2. Round
3. Square
4. Rectangle
5. Heart
6. Diamond
7. Oblong

Validates:
- Precise anatomical mask generation per shape.
- 100% horizontal symmetry of mask.
- 0.00 pixel change in eyes, eyebrows, nose, lips/mouth.
- Biometric identity preservation (similarity >= 0.95).
- Jawline symmetry delta < 60.0.
- Distinct structural transformation in cheeks, jawline, and chin.
"""

import os, sys, base64, io
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath("backend"))

from fastapi.testclient import TestClient
from main import app
from ai_services import BiometricEncoder, FaceIdentifier

client = TestClient(app)
encoder = BiometricEncoder()
identifier = FaceIdentifier()

FACE_SHAPES = [
    ("oval", "an oval face shape", "Oval"),
    ("round", "a round face shape", "Round"),
    ("square", "a square face shape", "Square"),
    ("rectangle", "a rectangle face shape", "Rectangle"),
    ("heart", "a heart-shaped face", "Heart"),
    ("diamond", "a diamond-shaped face with wide cheekbones", "Diamond"),
    ("oblong", "an oblong face shape", "Oblong")
]

def run_test_suite():
    print("=" * 80)
    print("F.A.C.E.S. 7-FACE-SHAPE REFINEMENT COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    base_path = "scratch/test_cf_initial_composite.png"
    orig_img = Image.open(base_path).convert("RGB")
    w, h = orig_img.size
    print(f"Loaded reference composite: {base_path} (Dimensions: {w}x{h})")

    buffered = io.BytesIO()
    orig_img.save(buffered, format="PNG")
    orig_b64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    v_orig = np.array(encoder.encode(orig_img))

    results = []

    for key, prompt, label in FACE_SHAPES:
        print(f"\n" + "-" * 70)
        print(f"Testing Face Shape: {label} (Prompt: '{prompt}')")
        print("-" * 70)

        # 1. Test Mask Generation directly
        mask = identifier.get_mask(orig_img, feature_name="faceshape", prompt=prompt)
        m_arr = np.array(mask)
        m_w, m_h = m_arr.shape[1], m_arr.shape[0]

        # Symmetry check on mask
        left_half = m_arr[:, :int(m_w / 2)]
        right_half_flipped = np.fliplr(m_arr[:, int(m_w / 2):])
        sym_err = int(np.max(np.abs(left_half - right_half_flipped)))
        cov_pct = float(np.mean(m_arr > 0) * 100)

        # Facial protection check on mask
        eyes_cov = int(np.max(m_arr[:int(m_h * 0.48), int(m_w * 0.18):int(m_w * 0.82)]))
        nose_cov = int(np.max(m_arr[int(m_h * 0.38):int(m_h * 0.64), int(m_w * 0.30):int(m_w * 0.70)]))
        mouth_cov = int(np.max(m_arr[int(m_h * 0.62):int(m_h * 0.77), int(m_w * 0.25):int(m_w * 0.75)]))

        print(f"[Mask] Coverage: {cov_pct:.1f}% | Symmetry Error: {sym_err} | Eyes: {eyes_cov} | Nose: {nose_cov} | Mouth: {mouth_cov}")
        assert sym_err == 0, f"Mask is not perfectly symmetrical! Error: {sym_err}"
        assert eyes_cov == 0, f"Mask covers eyes! Max: {eyes_cov}"
        assert nose_cov == 0, f"Mask covers nose! Max: {nose_cov}"
        assert mouth_cov == 0, f"Mask covers mouth! Max: {mouth_cov}"

        # 2. Call API endpoint /api/refine-feature with fixed seed 42
        payload = {
            "image": orig_b64,
            "feature_name": "faceshape",
            "refinement_prompt": prompt,
            "updated_prompt": f"forensic sketch pencil drawing of a human face, {prompt}",
            "seed": 42
        }

        print(f"[API] Sending refinement request to /api/refine-feature (seed=42)...")
        resp = client.post("/api/refine-feature", data=payload)
        assert resp.status_code == 200, f"API failed ({resp.status_code}): {resp.text}"

        res_data = resp.json()
        out_b64 = res_data["image"].split("base64,")[1]
        refined_img = Image.open(io.BytesIO(base64.b64decode(out_b64))).convert("RGB")
        out_path = f"scratch/test_shape_{key}.png"
        refined_img.save(out_path)
        print(f"[Result] Saved refined image to {out_path}")

        # 3. Analyze Output Quality & Symmetry
        ref_arr = np.array(refined_img.convert("L"), dtype=float)
        orig_arr = np.array(orig_img.convert("L"), dtype=float)

        # Jaw symmetry on refined image
        left_jaw = ref_arr[int(h * 0.55):int(h * 0.85), int(w * 0.15):int(w * 0.35)]
        right_jaw = ref_arr[int(h * 0.55):int(h * 0.85), int(w * 0.65):int(w * 0.85)]
        right_jaw_flipped = np.fliplr(right_jaw)
        jaw_sym_delta = float(np.mean(np.abs(left_jaw - right_jaw_flipped)))

        # Facial protection verification (pixels outside mask / internal core)
        diff = np.abs(ref_arr - orig_arr)
        mouth_diff = float(np.max(diff[int(h * 0.64):int(h * 0.76), int(w * 0.28):int(w * 0.72)]))
        nose_diff = float(np.max(diff[int(h * 0.42):int(h * 0.62), int(w * 0.32):int(w * 0.68)]))
        eyes_diff = float(np.max(diff[:int(h * 0.48), int(w * 0.20):int(w * 0.80)]))

        # Change in edited mask zone
        mask_512 = np.array(mask.resize((w, h)))
        mean_jaw_change = float(np.mean(diff[mask_512 > 10]))
        max_jaw_change = float(np.max(diff[mask_512 > 10]))

        # Biometric similarity
        v_ref = np.array(encoder.encode(refined_img))
        similarity = float(np.dot(v_orig, v_ref))

        print(f"[Metrics] Biometric Similarity: {similarity:.4f}")
        print(f"[Metrics] Jawline Symmetry Delta: {jaw_sym_delta:.2f} (Clean balanced shading)")
        print(f"[Metrics] Mean Change in Jaw/Cheek Area: {mean_jaw_change:.2f} px (Max: {max_jaw_change:.2f} px)")
        print(f"[Metrics] Max Change in Eyes: {eyes_diff:.2f} px (0.00 required)")
        print(f"[Metrics] Max Change in Nose: {nose_diff:.2f} px (0.00 required)")
        print(f"[Metrics] Max Change in Mouth: {mouth_diff:.2f} px (0.00 required)")

        assert similarity >= 0.95, f"Biometric similarity degraded below 0.95: {similarity:.4f}"
        assert mouth_diff == 0.0, f"Mouth changed! Diff: {mouth_diff}"
        assert nose_diff == 0.0, f"Nose changed! Diff: {nose_diff}"
        assert eyes_diff == 0.0, f"Eyes changed! Diff: {eyes_diff}"
        assert mean_jaw_change >= 4.0, f"Jawline did not transform enough: {mean_jaw_change:.2f} px"
        assert jaw_sym_delta <= 50.0, f"Jawline symmetry unbalanced: {jaw_sym_delta:.2f}"

        results.append({
            "shape": label,
            "similarity": similarity,
            "jaw_sym_delta": jaw_sym_delta,
            "jaw_change": mean_jaw_change,
            "mouth_diff": mouth_diff,
            "nose_diff": nose_diff,
            "eyes_diff": eyes_diff,
            "status": "PASS"
        })

    print("\n" + "=" * 80)
    print(f"{'Shape':12s} | {'Similarity':11s} | {'Jaw Sym Delta':14s} | {'Mean Change':12s} | {'Status':6s}")
    print("=" * 80)
    for r in results:
        print(f"{r['shape']:12s} | {r['similarity']:11.4f} | {r['jaw_sym_delta']:14.2f} | {r['jaw_change']:10.2f} px | {r['status']:6s}")
    print("=" * 80)
    print("ALL 7 FACE SHAPES PASSED CONSISTENCY AND IDENTITY PRESERVATION VERIFICATION!")

if __name__ == "__main__":
    run_test_suite()
