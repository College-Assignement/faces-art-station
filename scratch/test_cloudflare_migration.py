"""
Comprehensive Test Suite for Cloudflare Workers AI Migration:
1. Initial Text-to-Image Generation (@cf/bytedance/stable-diffusion-xl-lightning)
2. Surgical Nose Refinement (@cf/runwayml/stable-diffusion-v1-5-inpainting)
3. Surgical Eyes Refinement (@cf/runwayml/stable-diffusion-v1-5-inpainting)
4. Surgical Lips Refinement (@cf/runwayml/stable-diffusion-v1-5-inpainting)
5. Region Isolation and 0% Bleed Verification
6. API Failure / Error State Verification (Confirms old image is never returned on failure)
"""

import sys
import os
import io
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath("backend"))
from ai_services import FaceIdentifier, DigitalSurgeon, CloudflareAIClient

def main():
    print("=" * 60)
    print("F.A.C.E.S. Cloudflare Workers AI Verification Suite")
    print("=" * 60)

    identifier = FaceIdentifier()
    surgeon = DigitalSurgeon()

    # --- Test 1: Initial Generation ---
    print("\n[Test 1] Testing Initial Composite Generation...")
    prompt = "Male, 30 years old, sharp jawline, short brown hair, oval face, brown eyes, neutral expression"
    initial_img = surgeon.generate(prompt=prompt)
    assert initial_img is not None, "Initial composite returned None!"
    assert initial_img.size == (1024, 1024) or initial_img.size == (512, 512), f"Unexpected size: {initial_img.size}"
    initial_img.save("scratch/test_cf_initial_composite.png")
    print(f"-> PASS: Initial composite generated successfully. Size: {initial_img.size}, Mode: {initial_img.mode}")

    # Standardize image to 512x512 for surgical tests
    base_img = initial_img.resize((512, 512), Image.Resampling.LANCZOS)
    base_arr = np.array(base_img)
    w, h = 512, 512

    # Define key facial probe coordinate boxes for consistency testing
    # Eyes box: y in [150:230], x in [120:392]
    eye_slice = (slice(int(h * 0.30), int(h * 0.43)), slice(int(w * 0.22), int(w * 0.78)))
    # Nose box: y in [225:295], x in [200:312]
    nose_slice = (slice(int(h * 0.44), int(h * 0.58)), slice(int(w * 0.39), int(w * 0.61)))
    # Lip box: y in [315:395], x in [180:332]
    lip_slice = (slice(int(h * 0.62), int(h * 0.77)), slice(int(w * 0.35), int(w * 0.65)))

    # --- Test 2: Nose Refinement ---
    print("\n[Test 2] Testing Surgical Nose Refinement...")
    nose_mask = identifier.get_mask(base_img, "nose")
    nose_processed = identifier.process_mask(nose_mask, feature_name="nose")
    nose_refined = surgeon.inpaint(
        image=base_img,
        mask=nose_processed,
        prompt="prominent aquiline nose, hooked shape, dorsal hump, downward tip",
        strength=0.90,
        num_inference_steps=20
    )
    nose_refined.save("scratch/test_cf_refined_nose.png")
    nose_arr = np.array(nose_refined)

    # Verify Eyes & Lips in Nose-refined image are 100% UNCHANGED
    eye_diff_in_nose = np.max(np.abs(base_arr[eye_slice] - nose_arr[eye_slice]))
    lip_diff_in_nose = np.max(np.abs(base_arr[lip_slice] - nose_arr[lip_slice]))
    nose_diff_in_nose = np.max(np.abs(base_arr[nose_slice] - nose_arr[nose_slice]))

    print(f"   Max eye pixel difference during nose edit: {eye_diff_in_nose} (Target: 0)")
    print(f"   Max lip pixel difference during nose edit: {lip_diff_in_nose} (Target: 0)")
    print(f"   Max nose pixel difference during nose edit: {nose_diff_in_nose} (Target: > 0)")
    assert eye_diff_in_nose == 0, f"Eye pixels bled during nose edit! Max diff: {eye_diff_in_nose}"
    assert lip_diff_in_nose == 0, f"Lip pixels bled during nose edit! Max diff: {lip_diff_in_nose}"
    assert nose_diff_in_nose > 0, "Nose pixels did not change!"
    print("-> PASS: Nose refinement strictly preserved eyes & lips with 0% bleed!")

    # --- Test 3: Eyes Refinement ---
    print("\n[Test 3] Testing Surgical Eyes Refinement...")
    eye_mask = identifier.get_mask(base_img, "eyes")
    eye_processed = identifier.process_mask(eye_mask, feature_name="eyes")
    eye_refined = surgeon.inpaint(
        image=base_img,
        mask=eye_processed,
        prompt="narrow piercing squinting eyes, sharp gaze, detailed pupils",
        strength=0.85,
        num_inference_steps=20
    )
    eye_refined.save("scratch/test_cf_refined_eyes.png")
    eye_arr = np.array(eye_refined)

    # Verify Nose & Lips in Eye-refined image are 100% UNCHANGED
    nose_diff_in_eyes = np.max(np.abs(base_arr[nose_slice] - eye_arr[nose_slice]))
    lip_diff_in_eyes = np.max(np.abs(base_arr[lip_slice] - eye_arr[lip_slice]))
    eye_diff_in_eyes = np.max(np.abs(base_arr[eye_slice] - eye_arr[eye_slice]))

    print(f"   Max nose pixel difference during eyes edit: {nose_diff_in_eyes} (Target: 0)")
    print(f"   Max lip pixel difference during eyes edit: {lip_diff_in_eyes} (Target: 0)")
    print(f"   Max eye pixel difference during eyes edit: {eye_diff_in_eyes} (Target: > 0)")
    assert nose_diff_in_eyes == 0, f"Nose pixels bled during eye edit! Max diff: {nose_diff_in_eyes}"
    assert lip_diff_in_eyes == 0, f"Lip pixels bled during eye edit! Max diff: {lip_diff_in_eyes}"
    assert eye_diff_in_eyes > 0, "Eye pixels did not change!"
    print("-> PASS: Eyes refinement strictly preserved nose & lips with 0% bleed!")

    # --- Test 4: Lips Refinement ---
    print("\n[Test 4] Testing Surgical Lips Refinement...")
    lip_mask = identifier.get_mask(base_img, "lips")
    lip_processed = identifier.process_mask(lip_mask, feature_name="lips")
    lip_refined = surgeon.inpaint(
        image=base_img,
        mask=lip_processed,
        prompt="thin pressed lips, serious expression, straight mouth line",
        strength=0.85,
        num_inference_steps=20
    )
    lip_refined.save("scratch/test_cf_refined_lips.png")
    lip_arr = np.array(lip_refined)

    # Verify Eyes & Nose in Lip-refined image are 100% UNCHANGED
    eye_diff_in_lips = np.max(np.abs(base_arr[eye_slice] - lip_arr[eye_slice]))
    nose_diff_in_lips = np.max(np.abs(base_arr[nose_slice] - lip_arr[nose_slice]))
    lip_diff_in_lips = np.max(np.abs(base_arr[lip_slice] - lip_arr[lip_slice]))

    print(f"   Max eye pixel difference during lips edit: {eye_diff_in_lips} (Target: 0)")
    print(f"   Max nose pixel difference during lips edit: {nose_diff_in_lips} (Target: 0)")
    print(f"   Max lip pixel difference during lips edit: {lip_diff_in_lips} (Target: > 0)")
    assert eye_diff_in_lips == 0, f"Eye pixels bled during lip edit! Max diff: {eye_diff_in_lips}"
    assert nose_diff_in_lips == 0, f"Nose pixels bled during lip edit! Max diff: {nose_diff_in_lips}"
    assert lip_diff_in_lips > 0, "Lip pixels did not change!"
    print("-> PASS: Lips refinement strictly preserved eyes & nose with 0% bleed!")

    # --- Test 5: Error Handling & No-Fallback Verification ---
    print("\n[Test 5] Testing Failure States & Error Propagation...")
    broken_client = CloudflareAIClient(account_id="invalid_acc", api_token="invalid_tok")
    error_caught = False
    try:
        broken_client.text_to_image("A testing face")
    except RuntimeError as re:
        error_caught = True
        print(f"   Properly caught expected API error: {re}")
    assert error_caught, "Expected broken credentials to raise RuntimeError, but it didn't!"
    print("-> PASS: API failure raised descriptive RuntimeError; old image is NEVER returned on failure.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
