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
url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_inpaint}"

# Load initial image from test_cf_t2i.png
img = Image.open("scratch/test_cf_t2i.png").convert("RGB").resize((512, 512))

# Mask: Nose box (white = 255 inpaint, black = 0 keep)
w, h = 512, 512
mask = Image.new("L", (w, h), 0)
d = ImageDraw.Draw(mask)
d.rounded_rectangle([w * 0.36, h * 0.44, w * 0.64, h * 0.59], radius=12, fill=255)

# PNG bytes
img_buf = io.BytesIO()
img.save(img_buf, format="PNG")
img_bytes = img_buf.getvalue()

mask_buf = io.BytesIO()
mask.save(mask_buf, format="PNG")
mask_bytes = mask_buf.getvalue()

img_b64 = base64.b64encode(img_bytes).decode("utf-8")
mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

print(f"Image bytes: {len(img_bytes)}, Mask bytes: {len(mask_bytes)}")

# Let's test combinations:
configs = [
    {"name": "image_b64 + mask_image (b64)", "payload": {
        "prompt": "forensic sketch pencil drawing of a hooked nose",
        "image_b64": img_b64,
        "mask_image": mask_b64,
        "num_steps": 20,
        "strength": 0.85
    }},
    {"name": "image_b64 + mask (list of ints)", "payload": {
        "prompt": "forensic sketch pencil drawing of a hooked nose",
        "image_b64": img_b64,
        "mask": list(mask_bytes),
        "num_steps": 20,
        "strength": 0.85
    }},
    {"name": "image_b64 + mask_image (list of ints)", "payload": {
        "prompt": "forensic sketch pencil drawing of a hooked nose",
        "image_b64": img_b64,
        "mask_image": list(mask_bytes),
        "num_steps": 20,
        "strength": 0.85
    }},
    {"name": "image (list) + mask (list)", "payload": {
        "prompt": "forensic sketch pencil drawing of a hooked nose",
        "image": list(img_bytes),
        "mask": list(mask_bytes),
        "num_steps": 20,
        "strength": 0.85
    }},
]

for cfg in configs:
    name = cfg["name"]
    payload = cfg["payload"]
    print(f"\n--- Testing: {name} ---")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
        if r.status_code == 200:
            print(f"SUCCESS! Output size: {len(r.content)} bytes")
            with open(f"scratch/test_inpaint_success_{name[:10].replace(' ', '_')}.png", "wb") as f:
                f.write(r.content)
            break
        else:
            print(f"Response: {r.text[:250]}")
    except Exception as e:
        print(f"Error: {e}")

