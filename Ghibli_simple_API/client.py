import requests
import json  
from requests_toolbelt.multipart.encoder import MultipartEncoder

API_URL = "http://localhost:7860/generate"

def send_generation_request(image_path: str, output_path: str = "output.png", **kwargs):
    # Construct nested JSON request body with default values
    json_payload = {
        "prompt": "Ghibli Studio style, Charming hand-drawn anime-style illustration",
        "height": 768,
        "width": 768,
        "seed": 42,
        "control_type": "Ghibli",
        "guidance_scale": 3.5,
        "num_inference_steps": 25,
        "max_sequence_length": 512
    }
    
    # Update with any provided parameters
    json_payload.update(kwargs)

    # Use MultipartEncoder to precisely construct request body
    with open(image_path, "rb") as f:
        m = MultipartEncoder(
            fields={
                "request": ("request", json.dumps(json_payload), "application/json"),  # Key fix
                "spatial_image": (f.name, f, "image/png")
            }
        )

        try:
            response = requests.post(
                API_URL,
                data=m,
                headers={"Content-Type": m.content_type},  # Must manually set Content-Type
                timeout=60
            )

            if response.status_code == 200:
                with open(output_path, "wb") as out_f:
                    out_f.write(response.content)
                print("✅ Generation successful!")
            else:
                print(f"❌ Error code {response.status_code}")
                print("Detailed error:", response.json())

        except requests.exceptions.RequestException as e:
            print(f"🛑 Network error: {str(e)}")
        except json.JSONDecodeError:
            print("🛑 Invalid JSON response")

if __name__ == "__main__":
    # Example with default parameters
    send_generation_request("/workspace/EasyControl_Ghibli/test_imgs/03.png")
    
    # Example with custom parameters
    # send_generation_request(
    #     "/workspace/EasyControl_Ghibli/test_imgs/03.png",
    #     "custom_output.png",
    #     num_inference_steps=50,
    #     guidance_scale=4.0,
    #     seed=123
    # )