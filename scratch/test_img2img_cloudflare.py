import os
import requests
import base64
from PIL import Image
from dotenv import load_dotenv

project_root = r"c:\important\faces-art-station"
load_dotenv(os.path.join(project_root, ".env"))

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

with open("scratch/test_cf_t2i.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# Test 1: SDXL Lightning with image_b64 and strength (img2img)
url_sdxl = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/bytedance/stable-diffusion-xl-lightning"
print("\n--- Testing SDXL Lightning for img2img ---")
payload_sdxl_i2i = {
    "prompt": "forensic sketch pencil drawing of a human face, black and white",
    "image_b64": img_b64,
    "strength": 0.7
}
r1 = requests.post(url_sdxl, headers=headers, json=payload_sdxl_i2i, timeout=30)
print(f"SDXL img2img status: {r1.status_code}, content-type: {r1.headers.get('content-type')}")
if r1.status_code == 200:
    print(f"SDXL img2img SUCCESS! Bytes: {len(r1.content)}")
    with open("scratch/test_sdxl_img2img.png", "wb") as f:
        f.write(r1.content)
else:
    print(r1.text[:250])

# Test 2: Inpainting model without mask (does it do img2img?)
url_inpaint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/runwayml/stable-diffusion-v1-5-inpainting"
print("\n--- Testing inpainting model with image_b64 and strength (no mask) ---")
payload_inpaint_i2i = {
    "prompt": "forensic sketch pencil drawing of a human face, black and white",
    "image_b64": img_b64,
    "strength": 0.7
}
r2 = requests.post(url_inpaint, headers=headers, json=payload_inpaint_i2i, timeout=30)
print(f"Inpainting model without mask status: {r2.status_code}")
if r2.status_code == 200:
    print(f"Inpainting model img2img SUCCESS! Bytes: {len(r2.content)}")
    with open("scratch/test_inpaint_img2img.png", "wb") as f:
        f.write(r2.content)
else:
    print(r2.text[:250])

