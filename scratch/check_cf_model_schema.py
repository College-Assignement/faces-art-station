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

# Query model info
url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?search=stable-diffusion"
r = requests.get(url, headers=headers)
print("Search status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    for m in data.get("result", []):
        print(f"\nModel: {m.get('name')}")
        print("Description:", m.get("description"))
        print("Task:", m.get("task"))
        properties = m.get("properties", {})
        print("Properties:", properties)

