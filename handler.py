import runpod
import os
import json
import base64
from PIL import Image
import io
from app import ImageProcessor, single_condition_generate_image, multi_condition_generate_image

# 모델 초기화
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "EasyControl/models"
style_lora_base_path = "Shakker-Labs"
processor = ImageProcessor(base_path)

def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    if base64_string is None:
        return None
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error converting base64 to image: {str(e)}")
        return None

def image_to_base64(image):
    """PIL Image를 Base64 문자열로 변환"""
    if image is None:
        return None
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None

def handler(event):
    """runpod.io serverless handler"""
    try:
        # 입력 데이터 파싱
        input_data = event["input"]
        
        # 필수 파라미터 확인
        required_params = ["prompt", "height", "width", "seed"]
        for param in required_params:
            if param not in input_data:
                return {
                    "error": f"Missing required parameter: {param}"
                }
        
        # 이미지 데이터 변환
        subject_img = base64_to_image(input_data.get("subject_img"))
        spatial_img = base64_to_image(input_data.get("spatial_img"))
        
        # 단일 조건 생성
        if "control_type" in input_data:
            result = single_condition_generate_image(
                prompt=input_data["prompt"],
                subject_img=subject_img,
                spatial_img=spatial_img,
                height=input_data["height"],
                width=input_data["width"],
                seed=input_data["seed"],
                control_type=input_data["control_type"],
                style_lora=input_data.get("style_lora", "None")
            )
        # 다중 조건 생성
        else:
            result = multi_condition_generate_image(
                prompt=input_data["prompt"],
                subject_img=subject_img,
                spatial_img=spatial_img,
                height=input_data["height"],
                width=input_data["width"],
                seed=input_data["seed"]
            )
        
        # 결과 이미지를 base64로 변환
        result_base64 = image_to_base64(result)
        
        return {
            "status": "success",
            "image": result_base64
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }

# runpod.io handler 등록
runpod.serverless.start({"handler": handler}) 
