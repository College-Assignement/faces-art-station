import os
import requests
import io
import base64
from PIL import Image, ImageDraw
from dotenv import load_dotenv

project_root = r"c:\important\faces-art-station"
load_dotenv(os.path.join(project_root, ".env"))

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

model_inpaint = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
url_inpaint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_inpaint}"

# Load test image generated previously
test_img_path = "scratch/test_cf_t2i.png"
if not os.path.exists(test_img_path):
    # create dummy
    img = Image.new("RGB", (512, 512), color="lightgray")
    img.save(test_img_path)

img = Image.open(test_img_path).convert("RGB").resize((512, 512))

# Create a nose mask (white = area to inpaint, black = area to keep)
w, h = 512, 512
mask = Image.new("L", (w, h), 0)
d = ImageDraw.Draw(mask)
d.rounded_rectangle([w * 0.36, h * 0.44, w * 0.64, h * 0.59], radius=12, fill=255)

# Convert to PNG bytes
img_buf = io.BytesIO()
img.save(img_buf, format="PNG")
img_bytes = img_buf.getvalue()

mask_buf = io.BytesIO()
mask.save(mask_buf, format="PNG")
mask_bytes = mask_buf.getvalue()

# Test Format A: Array of integers (uint8 array)
print("\n--- Testing Format A: Array of uint8 integers ---")
payload_a = {
    "prompt": "forensic sketch pencil drawing of a hooked aquiline nose, black and white line art",
    "image": list(img_bytes),
    "mask": list(mask_bytes),
    "num_steps": 20,
    "strength": 0.85,
    "guidance": 7.5
}

try:
    r = requests.post(url_inpaint, headers=headers, json=payload_a, timeout=40)
    print(f"Format A Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
    if r.status_code == 200:
        print(f"Format A Success! Result size: {len(r.content)} bytes")
        with open("scratch/test_cf_inpaint_format_a.png", "wb") as f:
            f.write(r.content)
        print("Saved scratch/test_cf_inpaint_format_a.png")
    else:
        print(f"Format A Response: {r.text[:300]}")
except Exception as e:
    print(f"Format A Error: {e}")

# Test Format B: Base64 strings (image_b64)
print("\n--- Testing Format B: Base64 strings ---")
payload_b = {
    "prompt": "forensic sketch pencil drawing of a hooked aquiline nose, black and white line art",
    "image_b64": base64.b64encode(img_bytes).decode("utf-8"),
    "mask_b64": base64.b64encode(mask_bytes).decode("utf-8"),
    "num_steps": 20,
    "strength": 0.85,
    "guidance": 7.5
}

try:
    r = requests.post(url_inpaint, headers=headers, json=payload_b, timeout=40)
    print(f"Format B Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
    if r.status_code == 200:
        print(f"Format B Success! Result size: {len(r.content)} bytes")
        with open("scratch/test_cf_inpaint_format_b.png", "wb") as f:
            f.write(r.content)
        print("Saved scratch/test_cf_inpaint_format_b.png")
    else:
        print(f"Format B Response: {r.text[:300]}")
except Exception as e:
    print(f"Format B Error: {e}")

