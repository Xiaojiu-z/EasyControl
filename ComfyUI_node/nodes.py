import torch
import os
import sys
import numpy as np
from PIL import Image
import folder_paths
from huggingface_hub import hf_hub_download
from comfy.model_management import get_torch_device


# Get the absolute path of the directory containing the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add yuxuan_workflow to Python path
sys.path.append(os.path.join(current_dir, "easy_control"))

# Set model directory path
MODELS_PATH = os.path.join(current_dir, "models")
# Ensure model directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

try:
    # Attempt to import
    # Import FLUX related components
    from src.pipeline import FluxPipeline
    from src.transformer_flux import FluxTransformer2DModel
    from src.lora_helper import set_single_lora, set_multi_lora
except Exception as e:

    raise e

# Function for cache clearing
def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# 1. Base Model Loader Node
class EasyControlBaseModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
            }
        }
    
    RETURN_TYPES = ("EASY_CONTROL_PIPE", "EASY_CONTROL_TRANSFORMER")
    RETURN_NAMES = ("pipe", "transformer")
    FUNCTION = "load_model"
    CATEGORY = "EasyControl"

    def load_model(self, base_model_path):
        device = get_torch_device()
        
        # Initialize model
        pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device=device)
        transformer = FluxTransformer2DModel.from_pretrained(
            base_model_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            device=device
        )
        pipe.transformer = transformer
        pipe.to(device)
        
        return (pipe, transformer)

# New Node: FLUX Style LoRA Loader
class EasyControlLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("EASY_CONTROL_PIPE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "weight_name": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("EASY_CONTROL_PIPE",)
    FUNCTION = "load_lora"
    CATEGORY = "EasyControl"

    def load_lora(self, pipe, lora_name, lora_weight, weight_name=""):
        # Get the full path of the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        # Load LoRA weights
        print(f"Loading FLUX Style LoRA: {lora_name}, Weight: {lora_weight}")
        
        # If weight_name is empty, use lora_name as weight_name
        if not weight_name:
            weight_name = lora_name
            
        # Load LoRA weights
        pipe.load_lora_weights(lora_path, weight_name=weight_name)
        
        # Fuse LoRA
        pipe.fuse_lora(lora_weights=[lora_weight])
        
        # Ensure model is on device
        device = get_torch_device()
        pipe.to(device)
        
        return (pipe,)

# 2. Control Model Selector Node
class EasyControlModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_type": (["canny", "depth", "hedsketch", "pose", "seg", "inpainting", "subject"], {"default": "canny"}),
                "transformer": ("EASY_CONTROL_TRANSFORMER",),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "cond_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "use_plugin_path": ("BOOLEAN", {"default": True, "label": "Use plugin directory"}),
            }
        }
    
    RETURN_TYPES = ("EASY_CONTROL_TRANSFORMER",)
    FUNCTION = "select_control_model"
    CATEGORY = "EasyControl"

    def select_control_model(self, control_type, transformer, lora_weight, cond_size, use_plugin_path):
        # Select model directory
        if use_plugin_path:
            # Use models folder in plugin directory
            lora_path = MODELS_PATH
        else:
            # Use ComfyUI default models/easycontrol directory
            lora_path = os.path.join(folder_paths.models_dir, "easycontrol")
            os.makedirs(lora_path, exist_ok=True)
        
        # Control model path mapping
        control_models = {
            "canny": f"{lora_path}/canny.safetensors",
            "depth": f"{lora_path}/depth.safetensors",
            "hedsketch": f"{lora_path}/hedsketch.safetensors",
            "pose": f"{lora_path}/pose.safetensors",
            "seg": f"{lora_path}/seg.safetensors",
            "inpainting": f"{lora_path}/inpainting.safetensors",
            "subject": f"{lora_path}/subject.safetensors",
        }
        
        # If model file doesn't exist, try to download from HuggingFace
        if not os.path.exists(control_models[control_type]):
            print(f"Downloading {control_type} model to {lora_path}...")
            hf_hub_download(
                repo_id="Xiaojiu-Z/EasyControl", 
                filename=f"models/{control_type}.safetensors", 
                local_dir=lora_path
            )
        
        # Load control model
        set_single_lora(transformer, control_models[control_type], lora_weights=[lora_weight], cond_size=cond_size)
        
        return (transformer,)

# 3. Condition Image Upload Node
class EasyControlSpatialImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["stretch", "center", "pad"], {"default": "stretch"})
            }
        }
    
    RETURN_TYPES = ("EASY_CONTROL_SPATIAL_IMAGE",)
    FUNCTION = "prepare_spatial_image"
    CATEGORY = "EasyControl"

    def prepare_spatial_image(self, image, resize_mode):
        # Convert ComfyUI format to PIL format
        image_np = 255. * image.cpu().numpy()
        image_pil = Image.fromarray(np.clip(image_np[0], 0, 255).astype(np.uint8))
        
        # Return PIL image for subsequent processing
        return (image_pil,)

# 4. Text Encode Node
class EasyControlTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("EASY_CONTROL_PROMPT",)
    FUNCTION = "encode_text"
    CATEGORY = "EasyControl"

    def encode_text(self, text):
        # Here only pass the text, actual encoding will be done in the sampler
        # Because FLUX model may have its own text processing method
        return (text,)

# 5. EasyControl Sampler Node
class EasyControlSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("EASY_CONTROL_PIPE",),
                "transformer": ("EASY_CONTROL_TRANSFORMER",),
                "prompt": ("EASY_CONTROL_PROMPT",),
                "spatial_image": ("EASY_CONTROL_SPATIAL_IMAGE",),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_sequence_length": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "cond_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "EasyControl"

    def sample(self, pipe, transformer, prompt, spatial_image, height, width, guidance_scale, 
               num_inference_steps, seed, max_sequence_length, cond_size):
        # Set generator seed
        generator = torch.Generator("cpu").manual_seed(seed)
        
        # Execute image generation
        output = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
            spatial_images=[spatial_image],
            subject_images=[],
            cond_size=cond_size,
        )
        
        # Clear cache
        clear_cache(transformer)
        
        # Get generated image and convert to ComfyUI format
        image = output.images[0]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return (image_tensor,)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "EasyControlBaseModelLoader": EasyControlBaseModelLoader,
    "EasyControlLoraLoader": EasyControlLoraLoader,
    "EasyControlModelSelector": EasyControlModelSelector,
    "EasyControlSpatialImage": EasyControlSpatialImage,
    "EasyControlTextEncode": EasyControlTextEncode,
    "EasyControlSampler": EasyControlSampler,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyControlBaseModelLoader": "EasyControl Base Model Loader",
    "EasyControlLoraLoader": "EasyControl LoRA Loader",
    "EasyControlModelSelector": "EasyControl Control Model Selector",
    "EasyControlSpatialImage": "EasyControl Condition Image Preparation",
    "EasyControlTextEncode": "EasyControl Text Encoder",
    "EasyControlSampler": "EasyControl Sampler",
}