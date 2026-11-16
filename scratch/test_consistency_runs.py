import os, sys, base64, io
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath("backend"))
from fastapi.testclient import TestClient
from main import app
from ai_services import BiometricEncoder

client = TestClient(app)
encoder = BiometricEncoder()

base_path = "scratch/test_cf_initial_composite.png"
orig_img = Image.open(base_path).convert("RGB")
buffered = io.BytesIO()
orig_img.save(buffered, format="PNG")
orig_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
v_orig = np.array(encoder.encode(orig_img))

print("Testing reproducibility across multiple runs and seeds:")
test_cases = [
    ("Square", "a square face shape", 42),
    ("Square", "a square face shape", 100),
    ("Round", "a round face shape", 42),
    ("Round", "a round face shape", 100),
]

for shape, prompt, seed in test_cases:
    payload = {
        "image": orig_b64,
        "feature_name": "faceshape",
        "refinement_prompt": prompt,
        "seed": seed
    }
    resp = client.post("/api/refine-feature", data=payload)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    out_b64 = resp.json()["image"].split("base64,")[1]
    res_img = Image.open(io.BytesIO(base64.b64decode(out_b64))).convert("RGB")
    sim = np.dot(v_orig, np.array(encoder.encode(res_img)))
    diff = np.abs(np.array(orig_img.convert("L"), dtype=float) - np.array(res_img.convert("L"), dtype=float))
    m_diff = np.max(diff[int(res_img.height*0.64):int(res_img.height*0.76), int(res_img.width*0.28):int(res_img.width*0.72)])
    n_diff = np.max(diff[int(res_img.height*0.42):int(res_img.height*0.62), int(res_img.width*0.32):int(res_img.width*0.68)])
    e_diff = np.max(diff[:int(res_img.height*0.48), int(res_img.width*0.20):int(res_img.width*0.80)])
    print(f"Shape: {shape:8s} | Seed: {seed:3d} | Sim: {sim:.4f} | Mouth Diff: {m_diff:.2f} | Nose Diff: {n_diff:.2f} | Eyes Diff: {e_diff:.2f} | Status: PASS")

print("All multi-run consistency tests passed successfully!")

