import httpx
import base64
import io
import numpy as np
from PIL import Image, ImageDraw

def test_api_endpoint():
    print("Testing /api/refine-feature endpoint directly...")
    
    # 1. Create a 512x512 synthetic test image
    img = Image.new("RGB", (512, 512), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 75, 412, 435], outline=(50, 50, 50), width=3)
    # Nose
    draw.line([256, 200, 256, 270], fill=(50, 50, 50), width=2)
    draw.arc([230, 250, 280, 280], start=0, end=180, fill=(40, 40, 40), width=2)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    
    data = {
        "image": img_b64,
        "feature_name": "nose",
        "refinement_prompt": "small button nose with round tip",
        "updated_prompt": "forensic sketch of a suspect with a small button nose",
        "seed": "42"
    }
    
    with httpx.Client(timeout=60.0) as client:
        resp = client.post("http://localhost:8000/api/refine-feature", data=data)
        print("HTTP Status:", resp.status_code)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        
        result_json = resp.json()
        assert "image" in result_json, "Response missing 'image'"
        assert "message" in result_json, "Response missing 'message'"
        assert "debug_mask" in result_json, "Response missing 'debug_mask'"
        
        print("Message:", result_json["message"])
        
        # Verify returned image
        raw_b64 = result_json["image"].split("base64,")[1]
        out_img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
        out_arr = np.array(out_img).astype(float)
        orig_arr = np.array(img).astype(float)
        
        # Verify mask
        raw_mask_b64 = result_json["debug_mask"].split("base64,")[1]
        mask_img = Image.open(io.BytesIO(base64.b64decode(raw_mask_b64))).convert("L")
        mask_arr = np.array(mask_img)
        
        # Check unmasked preservation
        unmasked = (mask_arr == 0)
        unmasked_mae = np.mean(np.abs(orig_arr[unmasked] - out_arr[unmasked]))
        
        # Check nose change
        nose_area = (mask_arr > 128)
        nose_mae = np.mean(np.abs(orig_arr[nose_area] - out_arr[nose_area]))
        
        print(f"Endpoint Test Unmasked MAE: {unmasked_mae:.6f} (Must be 0.000000)")
        print(f"Endpoint Test Nose MAE:     {nose_mae:.6f} (Must be > 0)")
        
        assert unmasked_mae == 0.0, f"Unmasked area modified: MAE={unmasked_mae}"
        assert nose_mae > 0.0, f"Nose was not modified: MAE={nose_mae}"
        print(">>> ENDPOINT END-TO-END TEST PASSED! <<<")

if __name__ == "__main__":
    test_api_endpoint()
