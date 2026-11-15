import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend directory is in path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, backend_dir)

from ai_services import FaceIdentifier, DigitalSurgeon

def create_synthetic_forensic_sketch(width=512, height=512) -> Image.Image:
    """Create a realistic synthetic pencil forensic sketch for deterministic testing."""
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Face contour
    draw.ellipse([width * 0.20, height * 0.15, width * 0.80, height * 0.85], outline=(50, 50, 50), width=3)
    # Hairline
    draw.arc([width * 0.20, height * 0.10, width * 0.80, height * 0.45], start=180, end=360, fill=(30, 30, 30), width=4)
    # Left eye (y: 0.20 to 0.28, x: 0.28 to 0.42)
    draw.ellipse([width * 0.28, height * 0.22, width * 0.42, height * 0.28], outline=(40, 40, 40), width=2)
    draw.ellipse([width * 0.33, height * 0.23, width * 0.37, height * 0.27], fill=(20, 20, 20))
    # Right eye (y: 0.20 to 0.28, x: 0.58 to 0.72)
    draw.ellipse([width * 0.58, height * 0.22, width * 0.72, height * 0.28], outline=(40, 40, 40), width=2)
    draw.ellipse([width * 0.63, height * 0.23, width * 0.67, height * 0.27], fill=(20, 20, 20))
    # Eyebrows (y: 0.18)
    draw.line([width * 0.26, height * 0.19, width * 0.43, height * 0.20], fill=(30, 30, 30), width=3)
    draw.line([width * 0.57, height * 0.20, width * 0.74, height * 0.19], fill=(30, 30, 30), width=3)
    # Nose (center: y 0.38 to 0.58)
    draw.line([width * 0.50, height * 0.40, width * 0.50, height * 0.52], fill=(50, 50, 50), width=2)
    draw.arc([width * 0.45, height * 0.48, width * 0.55, height * 0.54], start=0, end=180, fill=(40, 40, 40), width=2)
    # Mouth (y: 0.70 to 0.75)
    draw.line([width * 0.38, height * 0.70, width * 0.62, height * 0.70], fill=(40, 40, 40), width=2)
    # Chin (y: 0.82)
    draw.arc([width * 0.45, height * 0.80, width * 0.55, height * 0.84], start=0, end=180, fill=(70, 70, 70), width=2)

    return img

def test_nose_refinement_preservation():
    print("=" * 70)
    print("TEST: Verifying Facial Region Preservation During Nose Refinement")
    print("=" * 70)

    # 1. Initialize AI Services
    identifier = FaceIdentifier()
    surgeon = DigitalSurgeon()

    # 2. Create base composite sketch
    orig_img = create_synthetic_forensic_sketch(512, 512)
    orig_arr = np.array(orig_img).astype(float)
    w, h = orig_img.size

    # 3. Get nose mask
    mask = identifier.get_mask(orig_img, "nose")
    processed_mask = identifier.process_mask(mask, blur_radius=3, dilate=True)
    mask_arr = np.array(processed_mask)

    print(f"Original image size: {orig_img.size}")
    print(f"Mask size: {processed_mask.size}")
    print(f"Unmasked pixel ratio (mask == 0): {(mask_arr == 0).mean():.4f}")
    print(f"Active inpainting pixel ratio (mask > 0): {(mask_arr > 0).mean():.4f}")

    # 4. Perform nose refinement (hooked aquiline nose)
    refinement_prompt = "Prominent aquiline nose, hooked shape, dorsal hump, downward tip"
    actual_prompt = (
        f"forensic sketch pencil drawing of a human nose, specifically {refinement_prompt}, "
        f"black and white line art, detailed pencil shading, clean contours, technical illustration style"
    )
    print(f"\nExecuting surgeon.inpaint for feature 'nose'...")
    refined_img = surgeon.inpaint(
        image=orig_img,
        mask=processed_mask,
        prompt=actual_prompt,
        negative_prompt="color, realistic photo, blur, double nose, deformed",
        seed=12345
    )

    refined_arr = np.array(refined_img).astype(float)

    # 5. Measure differences by region
    # A. Strictly unmasked pixels (where mask == 0)
    unmasked_indices = (mask_arr == 0)
    unmasked_mae = np.mean(np.abs(orig_arr[unmasked_indices] - refined_arr[unmasked_indices]))
    unmasked_max_diff = np.max(np.abs(orig_arr[unmasked_indices] - refined_arr[unmasked_indices]))

    # B. Forehead & Hair region (y: 0 to h*0.30)
    forehead_slice = slice(0, int(h * 0.30))
    forehead_mae = np.mean(np.abs(orig_arr[forehead_slice, :] - refined_arr[forehead_slice, :]))

    # C. Outer eyes & temples (y: h*0.20 to h*0.32, outside nose horizontally: x < w*0.28 or x > w*0.72)
    eyes_y_slice = slice(int(h * 0.20), int(h * 0.32))
    temple_left = slice(0, int(w * 0.28))
    temple_right = slice(int(w * 0.72), w)
    eyes_mae_left = np.mean(np.abs(orig_arr[eyes_y_slice, temple_left] - refined_arr[eyes_y_slice, temple_left]))
    eyes_mae_right = np.mean(np.abs(orig_arr[eyes_y_slice, temple_right] - refined_arr[eyes_y_slice, temple_right]))
    eyes_temples_mae = (eyes_mae_left + eyes_mae_right) / 2.0

    # D. Mouth & Chin region (y: h*0.70 to h)
    mouth_chin_slice = slice(int(h * 0.70), h)
    mouth_chin_mae = np.mean(np.abs(orig_arr[mouth_chin_slice, :] - refined_arr[mouth_chin_slice, :]))

    # E. Nose region (where mask is fully active, mask > 128)
    nose_indices = (mask_arr > 128)
    nose_mae = np.mean(np.abs(orig_arr[nose_indices] - refined_arr[nose_indices]))
    nose_max_diff = np.max(np.abs(orig_arr[nose_indices] - refined_arr[nose_indices]))

    print("\n" + "=" * 50)
    print("QUANTITATIVE RESULTS:")
    print("=" * 50)
    print(f"1. Forehead & Hair Region MAE: {forehead_mae:.6f} (Expected: 0.000000)")
    print(f"2. Eyes & Temples Region MAE:  {eyes_temples_mae:.6f} (Expected: 0.000000)")
    print(f"3. Mouth & Chin Region MAE:    {mouth_chin_mae:.6f} (Expected: 0.000000)")
    print(f"4. Strictly Unmasked Area MAE: {unmasked_mae:.6f} (Expected: 0.000000, Max Diff: {unmasked_max_diff})")
    print(f"5. Nose Inpainting Area MAE:   {nose_mae:.6f} (Expected: > 0, Max Diff: {nose_max_diff})")
    print("=" * 50)

    # Assertions
    assert unmasked_mae == 0.0, f"FAIL: Unmasked regions modified! MAE was {unmasked_mae}"
    assert forehead_mae == 0.0, f"FAIL: Forehead modified! MAE was {forehead_mae}"
    assert eyes_temples_mae == 0.0, f"FAIL: Eyes/Temples modified! MAE was {eyes_temples_mae}"
    assert mouth_chin_mae == 0.0, f"FAIL: Mouth/Chin modified! MAE was {mouth_chin_mae}"
    assert nose_mae > 0.0, f"FAIL: Nose was not modified! MAE was {nose_mae}"

    print("\n>>> ALL REGIONAL PRESERVATION ASSERTIONS PASSED SUCCESSFULLY! <<<")
    print("Only the nose was modified. All other facial regions remain 100% pixel-identical.")

if __name__ == "__main__":
    test_nose_refinement_preservation()
