import os
import requests
from dotenv import load_dotenv

project_root = r"c:\important\faces-art-station"
load_dotenv(os.path.join(project_root, ".env"))

account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/schema?model=@cf/runwayml/stable-diffusion-v1-5-inpainting"
r = requests.get(url, headers=headers)
print("Schema status:", r.status_code)
if r.status_code == 200:
    import json
    print(json.dumps(r.json(), indent=2))
else:
    print(r.text)

