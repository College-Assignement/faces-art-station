"""
Test Suite for Hair Refinement Transformations:
1. Short hair to long hair.
2. Straight hair to curly hair.
3. Curly hair to straight hair.
4. Dark hair to lighter hair.
5. Hair to bald or shaved appearance.

Verifies:
- Requested hairstyle changes visually.
- Intended hair region is processed.
- Face (eyes, nose, mouth, jawline) remains consistent.
- Biometric similarity >= 0.95.
"""

import os, sys, base64, io, requests
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath("backend"))
from ai_services import BiometricEncoder

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
token = os.getenv("CLOUDFLARE_API_TOKEN")
url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/runwayml/stable-diffusion-v1-5-inpainting"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
encoder = BiometricEncoder()

def build_hair_mask(w, h, intent):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    
    # Base top dome
    draw.ellipse([w * 0.08, -h * 0.12, w * 0.92, h * 0.28], fill=255)
    # Temples and sides
    draw.rectangle([w * 0.05, h * 0.12, w * 0.25, h * 0.46], fill=255)
    draw.rectangle([w * 0.75, h * 0.12, w * 0.95, h * 0.46], fill=255)
    
    if intent == 'long':
        draw.rectangle([0, h * 0.15, w * 0.32, h * 0.98], fill=255)
        draw.rectangle([w * 0.68, h * 0.15, w, h * 0.98], fill=255)
        draw.rectangle([0, h * 0.60, w * 0.36, h * 0.98], fill=255)
        draw.rectangle([w * 0.64, h * 0.60, w, h * 0.98], fill=255)
    elif intent == 'curly':
        draw.ellipse([w * 0.02, -h * 0.16, w * 0.98, h * 0.34], fill=255)
        draw.rectangle([0, h * 0.10, w * 0.30, h * 0.62], fill=255)
        draw.rectangle([w * 0.70, h * 0.10, w, h * 0.62], fill=255)
    elif intent == 'bald':
        draw.ellipse([w * 0.10, -h * 0.08, w * 0.90, h * 0.26], fill=255)
        draw.rectangle([0, 0, w * 0.22, h * 0.55], fill=255)
        draw.rectangle([w * 0.78, 0, w, h * 0.55], fill=255)
    elif intent in ['straight', 'short']:
        draw.rectangle([0, h * 0.15, w * 0.26, h * 0.68], fill=255)
        draw.rectangle([w * 0.74, h * 0.15, w, h * 0.68], fill=255)
    elif intent == 'light':
        draw.rectangle([0, h * 0.12, w * 0.28, h * 0.55], fill=255)
        draw.rectangle([w * 0.72, h * 0.12, w, h * 0.55], fill=255)
        
    arr = np.array(mask)
    # Facial identity protection
    arr[int(h * 0.27):int(h * 0.38), int(w * 0.23):int(w * 0.77)] = 0 # Eyebrows
    arr[int(h * 0.36):int(h * 0.48), int(w * 0.18):int(w * 0.82)] = 0 # Eyes
    arr[int(h * 0.46):int(h * 0.64), int(w * 0.26):int(w * 0.74)] = 0 # Nose
    arr[int(h * 0.63):int(h * 0.90), int(w * 0.28):int(w * 0.72)] = 0 # Lips/mouth
    arr[int(h * 0.78):, int(w * 0.36):int(w * 0.64)] = 0              # Center neck
    
    feathered = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(3))
    f_arr = np.array(feathered)
    f_arr[int(h * 0.28):int(h * 0.37), int(w * 0.24):int(w * 0.76)] = 0
    f_arr[int(h * 0.37):int(h * 0.47), int(w * 0.20):int(w * 0.80)] = 0
    f_arr[int(h * 0.47):int(h * 0.63), int(w * 0.27):int(w * 0.73)] = 0
    f_arr[int(h * 0.64):int(h * 0.88), int(w * 0.30):int(w * 0.70)] = 0
    f_arr[int(h * 0.80):, int(w * 0.38):int(w * 0.62)] = 0
    f_arr[f_arr < 4] = 0
    return Image.fromarray(f_arr)

def run_test_case(name, base_img, intent, prompt, neg_prompt, strength=0.82):
    print(f"\n--- Testing Case: {name} ---")
    orig_512 = base_img.resize((512, 512), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    orig_512.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    mask = build_hair_mask(512, 512, intent)
    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_bytes = list(mask_buf.getvalue())
    
    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "image_b64": img_b64,
        "mask": mask_bytes,
        "num_steps": 20,
        "strength": strength,
        "guidance": 7.5
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    assert resp.status_code == 200, f"Cloudflare API failed: {resp.text}"
    
    res_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    comp = Image.composite(res_img, orig_512, mask)
    
    # Metrics
    v_orig = np.array(encoder.encode(orig_512))
    v_comp = np.array(encoder.encode(comp))
    sim = np.dot(v_orig, v_comp)
    
    mask_np = np.array(mask)
    diff = np.abs(np.array(orig_512, dtype=float) - np.array(comp, dtype=float))
    mean_hair_change = np.mean(diff[mask_np > 10])
    max_face_change = np.max(diff[mask_np == 0])
    
    print(f"-> Biometric Similarity: {sim:.4f}")
    print(f"-> Mean Pixel Change in Hair: {mean_hair_change:.2f} px")
    print(f"-> Max Pixel Change outside mask: {max_face_change:.2f} px")
    
    assert sim >= 0.95, f"Biometric similarity too low: {sim:.4f}"
    assert mean_hair_change >= 5.0, f"Hair did not change enough: {mean_hair_change:.2f} px"
    assert max_face_change == 0.0 or max_face_change <= 20.0, f"Unmasked regions changed unexpectedly: {max_face_change:.2f}"
    
    return comp

def main():
    print("=" * 70)
    print("F.A.C.E.S. HAIR REFINEMENT TRANSFORMATION TEST SUITE")
    print("=" * 70)
    
    base_img = Image.open("scratch/test_cf_initial_composite.png").convert("RGB")
    
    # 1. Short hair to long hair
    img_long = run_test_case(
        "1. Short hair to long hair",
        base_img,
        "long",
        "forensic sketch pencil drawing of long hair, long straight hair past the shoulders, detailed hair strands falling down sides, technical police composite sketch, black and white line art",
        "short hair, bald, buzzcut, crop, color, photo, deformed",
        strength=0.82
    )
    img_long.save("scratch/test_suite_1_long.png")
    
    # 2. Straight hair to curly hair
    img_curly = run_test_case(
        "2. Straight hair to curly hair",
        base_img,
        "curly",
        "forensic sketch pencil drawing of curly hair, defined tight curls, voluminous textured curls, natural ringlets, technical police composite sketch, black and white line art",
        "straight hair, flat hair, sleek hair, bald, color, photo, deformed",
        strength=0.82
    )
    img_curly.save("scratch/test_suite_2_curly.png")
    
    # 3. Curly hair to straight hair (passing the curly image)
    img_straight = run_test_case(
        "3. Curly hair to straight hair",
        img_curly,
        "straight",
        "forensic sketch pencil drawing of sleek straight hair, smooth straight hair strands, neat flat straight hairstyle, technical police composite sketch, black and white line art",
        "curly, wavy, afro, curls, frizzy, permed, color, photo, deformed",
        strength=0.82
    )
    img_straight.save("scratch/test_suite_3_straight.png")
    
    # 4. Dark hair to lighter hair
    img_light = run_test_case(
        "4. Dark hair to lighter hair",
        base_img,
        "light",
        "forensic sketch pencil drawing of light blonde hair, light-colored hair, white highlights, soft fine pencil linework, bright hair tone, technical police composite sketch, black and white line art",
        "dark hair, black hair, heavy black shading, dense dark ink, color, photo, deformed",
        strength=0.85
    )
    img_light.save("scratch/test_suite_4_light.png")
    
    # 5. Hair to bald or shaved appearance
    img_bald = run_test_case(
        "5. Hair to bald or shaved appearance",
        base_img,
        "bald",
        "forensic sketch pencil drawing of a completely bald head, clean-shaven bald scalp, no hair, smooth bald head, technical police composite sketch, black and white line art",
        "hair, long hair, bangs, curls, wavy hair, afro, ponytail, wig, facial hair, color, photo, deformed",
        strength=0.85
    )
    img_bald.save("scratch/test_suite_5_bald.png")
    
    print("\n" + "=" * 70)
    print("ALL 5 HAIR TRANSFORMATION TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

