import sys
import os
import base64
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath("backend"))
from main import app

client = TestClient(app)

def test_endpoints():
    print("--- 1. Testing Root Endpoint ---")
    r0 = client.get("/")
    assert r0.status_code == 200, f"Root endpoint failed: {r0.status_code}"
    print("Root Status:", r0.json())

    print("\n--- 2. Testing /api/generate-composite Endpoint ---")
    gen_data = {
        "prompt": "Female, 28 years old, oval face, green eyes, wavy blonde hair, light skin",
        "seed": "0"
    }
    r1 = client.post("/api/generate-composite", data=gen_data)
    assert r1.status_code == 200, f"Generate failed: {r1.status_code}, {r1.text}"
    res1 = r1.json()
    assert "image" in res1 and res1["image"].startswith("data:image/png;base64,"), "Invalid image response"
    print("Generate Composite: SUCCESS! Image prefix:", res1["image"][:40])

    gen_img_b64 = res1["image"]

    print("\n--- 3. Testing /api/refine-feature (Nose) Endpoint ---")
    refine_data = {
        "image": gen_img_b64,
        "feature_name": "nose",
        "refinement_prompt": "hooked",
        "updated_prompt": "Female, 28 years old, oval face, green eyes, wavy blonde hair, hooked nose",
        "seed": "0"
    }
    r2 = client.post("/api/refine-feature", data=refine_data)
    assert r2.status_code == 200, f"Refine failed: {r2.status_code}, {r2.text}"
    res2 = r2.json()
    assert "image" in res2 and res2["image"].startswith("data:image/png;base64,"), "Invalid refine response"
    assert "debug_mask" in res2, "Missing debug_mask"
    assert res2["message"] == "Successfully refined nose"
    print("Refine Feature: SUCCESS! Message:", res2["message"])

    print("\n--- 4. Testing /api/search-database Endpoint ---")
    search_data = {
        "image": gen_img_b64
    }
    r3 = client.post("/api/search-database", data=search_data)
    assert r3.status_code == 200, f"Search failed: {r3.status_code}, {r3.text}"
    res3 = r3.json()
    assert "matches" in res3 and len(res3["matches"]) > 0, "No matches found in database"
    print(f"Search Database: SUCCESS! Found {len(res3['matches'])} matches. Top match score: {res3['matches'][0]['score']}%")

    print("\n>>> ALL API ENDPOINTS VERIFIED AND WORKING PERFECTLY! <<<")

if __name__ == "__main__":
    test_endpoints()

