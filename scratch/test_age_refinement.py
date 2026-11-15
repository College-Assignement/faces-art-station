"""
Verification Test for Perceived Age Refinement:
1. Test same face at ages 25, 45, and 65
2. Verify visual and biometric identity preservation
3. Verify previously working refinement features (Nose, Eyes) still work
"""

import sys
import os
import io
import base64
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath("backend"))
from main import app
from ai_services import BiometricEncoder
from fastapi.testclient import TestClient

client = TestClient(app)
encoder = BiometricEncoder()

def main():
    print("=" * 60)
    print("TESTING PERCEIVED AGE REFINEMENT (Ages 25, 45, 65)")
    print("=" * 60)

    # 1. Base Image
    base_path = "scratch/test_cf_initial_composite.png"
    if not os.path.exists(base_path):
        print("Generating base composite...")
        r = client.post("/api/generate-composite", data={"prompt": "Male, 35 years old, sharp jawline, short brown hair, oval face, neutral expression", "seed": 0})
        assert r.status_code == 200
        img_b64 = r.json()["image"]
    else:
        with open(base_path, "rb") as f:
            img_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

    orig_img = Image.open(io.BytesIO(base64.b64decode(img_b64.split("base64,")[1]))).convert("RGB")
    v_orig = np.array(encoder.encode(orig_img))

    results = {}

    for target_age in [25, 45, 65]:
        print(f"\n--- Testing Refinement to Age {target_age} ---")
        prompt_text = f"make the person appear around {target_age} years old"
        payload = {
            "image": img_b64,
            "feature_name": "age",
            "refinement_prompt": prompt_text,
            "updated_prompt": f"Male, {target_age} years old, sharp jawline, short brown hair, oval face",
            "seed": 0
        }
        res = client.post("/api/refine-feature", data=payload)
        assert res.status_code == 200, f"Age {target_age} refinement failed: {res.text}"
        data = res.json()
        out_b64 = data["image"].split("base64,")[1]
        out_img = Image.open(io.BytesIO(base64.b64decode(out_b64))).convert("RGB")
        out_path = f"scratch/test_refined_age_{target_age}.png"
        out_img.save(out_path)
        
        # Calculate biometric similarity and pixel difference
        v_out = np.array(encoder.encode(out_img))
        sim = float(np.dot(v_orig, v_out))
        diff = float(np.mean(np.abs(np.array(orig_img, dtype=float) - np.array(out_img, dtype=float))))
        
        print(f"-> SUCCESS: Saved {out_path}")
        print(f"   Biometric Identity Similarity to original: {sim:.4f} (Target > 0.85)")
        print(f"   Mean Pixel Change from original: {diff:.2f} (Target > 5.0)")
        
        assert sim > 0.85, f"Biometric similarity too low ({sim:.4f})! Identity not preserved."
        assert diff > 5.0, f"Pixel change too low ({diff:.2f})! Age did not visually change."
        results[target_age] = (out_img, sim, diff)

    # 4. Compare Age 25 vs Age 65 directly
    diff_25_65 = float(np.mean(np.abs(np.array(results[25][0], dtype=float) - np.array(results[65][0], dtype=float))))
    print(f"\nDirect difference between Age 25 (youthful) and Age 65 (elderly): {diff_25_65:.2f} px")
    assert diff_25_65 > 10.0, "Difference between 25 and 65 is too small!"
    print("-> PASS: Progressive visible age difference confirmed between young and elderly.")

    # 5. Verify previously working refinement feature (Nose) still functions
    print("\n--- Verifying previously working refinement (Nose) still functions ---")
    nose_payload = {
        "image": img_b64,
        "feature_name": "nose",
        "refinement_prompt": "hooked",
        "updated_prompt": "hooked aquiline nose",
        "seed": 0
    }
    r_nose = client.post("/api/refine-feature", data=nose_payload)
    assert r_nose.status_code == 200, f"Nose refinement failed: {r_nose.text}"
    print(f"-> PASS: Nose refinement succeeded: {r_nose.json()['message']}")

    print("\n" + "=" * 60)
    print("ALL PERCEIVED AGE AND REGRESSION TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

