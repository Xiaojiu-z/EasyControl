from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
import io
import json
import time
import torch
from PIL import Image
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

from safetensors.torch import save_file
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora, unset_lora

# Initialize FastAPI application
app = FastAPI(title="EasyControl API", description="API for Ghibli-style image generation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
base_path = "black-forest-labs/FLUX.1-dev"
lora_base_path = "../models"

pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
    base_path, 
    subfolder="transformer", 
    torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.to("cuda")

# Define request model
class GenerationRequest(BaseModel):
    prompt: str = Field(
        default="Ghibli Studio style, Charming hand-drawn anime-style illustration",
        description="Text prompt for image generation"
    )
    height: int = Field(
        default=768,
        ge=256,
        le=1024,
        description="Image height between 256 and 1024"
    )
    width: int = Field(
        default=768,
        ge=256,
        le=1024,
        description="Image width between 256 and 1024"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    control_type: str = Field(
        default="Ghibli",
        description="Control type (currently only supports Ghibli)"
    )
    guidance_scale: float = Field(
        default=3.5,
        gt=0,
        description="Classifier-free guidance scale (e.g. 3.5)"
    )
    num_inference_steps: int = Field(
        default=25,
        ge=1,
        description="Number of denoising steps (e.g. 25)"
    )
    max_sequence_length: int = Field(
        default=512,
        ge=64,
        description="Maximum token sequence length (e.g. 512)"
    )

# Define response model
class GenerationResponse(BaseModel):
    status: str
    image_base64: Optional[str] = None
    error: Optional[str] = None

def clear_cache(transformer):
    """Clear transformer cache"""
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

@app.post(
    "/generate", 
    response_model=GenerationResponse,
    responses={200: {"content": {"image/png": {}}}},
    description="Generate image with Ghibli style control"
)
async def generate_image(
    request: GenerationRequest,
    spatial_image: UploadFile = File(...)
):
    try:
        # Process input image
        image_data = await spatial_image.read()
        spatial_img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parameter validation
        if request.control_type != "Ghibli":
            raise ValueError("Only Ghibli control type is supported currently")
        
        if not (256 <= request.height <= 1024) or not (256 <= request.width <= 1024):
            raise ValueError("Width and height must be between 256 and 1024")
            
        if request.guidance_scale <= 0:
            raise ValueError("guidance_scale must be greater than 0")
            
        if request.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
            
        if request.max_sequence_length < 64:
            raise ValueError("max_sequence_length must be >= 64")

        # Set LoRA
        lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
        set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)

        # Generate image
        spatial_imgs = [spatial_img]
        generated_image = pipe(
            request.prompt,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(request.seed),
            subject_images=[],
            spatial_images=spatial_imgs,
            cond_size=512,
        ).images[0]

        clear_cache(pipe.transformer)

        # Convert image to byte stream
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)