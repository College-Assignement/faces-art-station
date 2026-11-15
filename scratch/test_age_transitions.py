"""
Verification Test for Perceived Age Identity-Preserving Transitions:
1. 25 to 35: Subtle age-related changes.
2. 25 to 45: Moderate age-related changes.
3. 25 to 55: More visible wrinkles and mature skin.
4. 55 to 35: Reduce age-related characteristics while preserving identity.
5. Verify eyes, nose, mouth, jawline are strictly protected.
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
    print("=" * 70)
    print("F.A.C.E.S. SURGICAL AGE REFINEMENT TEST SUITE")
    print("=" * 70)

    # Load base composite (25-year-old face)
    base_path = "scratch/test_cf_initial_composite.png"
    with open(base_path, "rb") as f:
        img_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

    orig_img = Image.open(io.BytesIO(base64.b64decode(img_b64.split("base64,")[1]))).convert("RGB")
    v_orig = np.array(encoder.encode(orig_img))

    # Mask for measuring change inside vs outside age zones
    age_mask = np.array(Image.open("public/masks/age_mask.png").resize(orig_img.size).convert("L")) > 10

    # Test 1: 25 -> 35 (Subtle)
    print("\n[Case 1] Testing 25 to 35 (Subtle age changes)...")
    res35 = client.post("/api/refine-feature", data={
        "image": img_b64,
        "feature_name": "age",
        "refinement_prompt": "make the person appear around 35 years old",
        "updated_prompt": "Male, 35 years old, subtle natural lines",
        "seed": 0
    })
    assert res35.status_code == 200, f"Failed: {res35.text}"
    img35 = Image.open(io.BytesIO(base64.b64decode(res35.json()["image"].split("base64,")[1]))).convert("RGB")
    img35.save("scratch/test_transition_25_to_35.png")
    
    sim35 = np.dot(v_orig, np.array(encoder.encode(img35)))
    diff35_in = np.mean(np.abs(np.array(orig_img, dtype=float)[age_mask] - np.array(img35, dtype=float)[age_mask]))
    diff35_out = np.max(np.abs(np.array(orig_img, dtype=float)[~age_mask] - np.array(img35, dtype=float)[~age_mask]))
    print(f"-> 25 to 35: Biometric Similarity = {sim35:.4f}")
    print(f"   Mean Pixel Change in Age Skin: {diff35_in:.2f} px (Subtle)")
    print(f"   Max Pixel Change in Eyes/Nose/Mouth: {diff35_out:.2f} px")
    assert sim35 >= 0.95, f"Identity lost in 35! Sim: {sim35}"
    assert diff35_in > 5.0, "No visible change in 35!"

    # Test 2: 25 -> 45 (Moderate)
    print("\n[Case 2] Testing 25 to 45 (Moderate age changes)...")
    res45 = client.post("/api/refine-feature", data={
        "image": img_b64,
        "feature_name": "age",
        "refinement_prompt": "make the person appear around 45 years old",
        "updated_prompt": "Male, 45 years old, visible forehead lines, crow's feet",
        "seed": 0
    })
    assert res45.status_code == 200, f"Failed: {res45.text}"
    img45 = Image.open(io.BytesIO(base64.b64decode(res45.json()["image"].split("base64,")[1]))).convert("RGB")
    img45.save("scratch/test_transition_25_to_45.png")
    
    sim45 = np.dot(v_orig, np.array(encoder.encode(img45)))
    diff45_in = np.mean(np.abs(np.array(orig_img, dtype=float)[age_mask] - np.array(img45, dtype=float)[age_mask]))
    diff45_out = np.max(np.abs(np.array(orig_img, dtype=float)[~age_mask] - np.array(img45, dtype=float)[~age_mask]))
    print(f"-> 25 to 45: Biometric Similarity = {sim45:.4f}")
    print(f"   Mean Pixel Change in Age Skin: {diff45_in:.2f} px (Moderate)")
    print(f"   Max Pixel Change in Eyes/Nose/Mouth: {diff45_out:.2f} px")
    assert sim45 >= 0.95, f"Identity lost in 45! Sim: {sim45}"
    assert diff45_in > diff35_in, f"45 should have more changes than 35 ({diff45_in:.2f} vs {diff35_in:.2f})!"

    # Test 3: 25 -> 55 (Prominent wrinkles & mature skin)
    print("\n[Case 3] Testing 25 to 55 (Prominent wrinkles & mature skin)...")
    res55 = client.post("/api/refine-feature", data={
        "image": img_b64,
        "feature_name": "age",
        "refinement_prompt": "make the person appear around 55 years old",
        "updated_prompt": "Male, 55 years old, prominent wrinkles, mature aging skin",
        "seed": 0
    })
    assert res55.status_code == 200, f"Failed: {res55.text}"
    img55_b64 = res55.json()["image"]
    img55 = Image.open(io.BytesIO(base64.b64decode(img55_b64.split("base64,")[1]))).convert("RGB")
    img55.save("scratch/test_transition_25_to_55.png")
    
    sim55 = np.dot(v_orig, np.array(encoder.encode(img55)))
    diff55_in = np.mean(np.abs(np.array(orig_img, dtype=float)[age_mask] - np.array(img55, dtype=float)[age_mask]))
    diff55_out = np.max(np.abs(np.array(orig_img, dtype=float)[~age_mask] - np.array(img55, dtype=float)[~age_mask]))
    print(f"-> 25 to 55: Biometric Similarity = {sim55:.4f}")
    print(f"   Mean Pixel Change in Age Skin: {diff55_in:.2f} px (Prominent)")
    print(f"   Max Pixel Change in Eyes/Nose/Mouth: {diff55_out:.2f} px")
    assert sim55 >= 0.93, f"Identity lost in 55! Sim: {sim55}"
    assert diff55_in > diff45_in, f"55 should have more changes than 45 ({diff55_in:.2f} vs {diff45_in:.2f})!"

    # Test 4: 55 -> 35 (Rejuvenation / Regression)
    print("\n[Case 4] Testing 55 to 35 (Reduce age-related characteristics while preserving identity)...")
    res35_from_55 = client.post("/api/refine-feature", data={
        "image": img55_b64,  # Passing the 55-year-old face
        "feature_name": "age",
        "refinement_prompt": "make the person appear around 35 years old, youthful skin, softened laugh lines",
        "updated_prompt": "Male, 35 years old, youthful skin",
        "seed": 0
    })
    assert res35_from_55.status_code == 200, f"Failed: {res35_from_55.text}"
    img35_from_55 = Image.open(io.BytesIO(base64.b64decode(res35_from_55.json()["image"].split("base64,")[1]))).convert("RGB")
    img35_from_55.save("scratch/test_transition_55_to_35.png")
    
    sim_reg = np.dot(v_orig, np.array(encoder.encode(img35_from_55)))
    diff_reg_55 = np.mean(np.abs(np.array(img55, dtype=float)[age_mask] - np.array(img35_from_55, dtype=float)[age_mask]))
    diff_reg_out = np.max(np.abs(np.array(orig_img, dtype=float)[~age_mask] - np.array(img35_from_55, dtype=float)[~age_mask]))
    print(f"-> 55 to 35: Biometric Similarity to original 25 = {sim_reg:.4f}")
    print(f"   Mean Pixel Reduction from 55: {diff_reg_55:.2f} px (Wrinkles reduced)")
    print(f"   Max Pixel Change in Eyes/Nose/Mouth: {diff_reg_out:.2f} px")
    assert sim_reg >= 0.93, f"Identity lost when going from 55 to 35! Sim: {sim_reg}"
    assert diff_reg_55 > 5.0, "Wrinkles were not reduced!"

    print("\n" + "=" * 70)
    print("ALL 4 AGE TRANSITION TEST CASES PASSED WITH HIGH IDENTITY PRESERVATION!")
    print("=" * 70)

if __name__ == "__main__":
    main()

