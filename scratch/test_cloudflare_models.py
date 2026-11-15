import os
import requests
from dotenv import load_dotenv

project_root = r"c:\important\faces-art-station"
load_dotenv(os.path.join(project_root, ".env"))

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

print(f"Cloudflare Account ID: {account_id[:6]}... (length: {len(account_id)})")
print(f"Cloudflare API Token: {api_token[:6]}... (length: {len(api_token)})")

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# 1. Test Text-to-Image model: @cf/bytedance/stable-diffusion-xl-lightning
model_t2i = "@cf/bytedance/stable-diffusion-xl-lightning"
url_t2i = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_t2i}"

print(f"\n1. Testing Text-to-Image: {model_t2i}")
payload_t2i = {
    "prompt": "forensic sketch pencil drawing of a 30s man, black and white portrait"
}

try:
    r = requests.post(url_t2i, headers=headers, json=payload_t2i, timeout=30)
    print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
    if r.status_code == 200:
        print(f"Success! Response bytes length: {len(r.content)}")
        with open("scratch/test_cf_t2i.png", "wb") as f:
            f.write(r.content)
        print("Saved scratch/test_cf_t2i.png")
    else:
        print(f"Response: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")

# 2. Check model availability for img2img and inpainting
for model_name in [
    "@cf/runwayml/stable-diffusion-v1-5-img2img",
    "@cf/runwayml/stable-diffusion-v1-5-inpainting",
    "@cf/bytedance/stable-diffusion-xl-lightning"
]:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
    print(f"\nChecking model endpoint availability: {model_name}")
    try:
        # Send minimal probe
        r = requests.post(url, headers=headers, json={"prompt": "test probe"}, timeout=15)
        print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
        if r.status_code != 200:
            print(f"Response: {r.text[:300]}")
        else:
            print(f"Response size: {len(r.content)} bytes")
    except Exception as e:
        print(f"Error: {e}")

