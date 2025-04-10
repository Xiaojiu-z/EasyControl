import torch
import os
import sys
import numpy as np
from PIL import Image
import folder_paths
from huggingface_hub import hf_hub_download
from comfy.model_management import get_torch_device


# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将yuxuan_workflow添加到Python路径
sys.path.append(os.path.join(current_dir, "easy_control"))

# 设置模型目录路径
MODELS_PATH = os.path.join(current_dir, "models")
# 确保模型目录存在
os.makedirs(MODELS_PATH, exist_ok=True)

try:
    # 尝试导入
    # 导入FLUX相关组件
    from src.pipeline import FluxPipeline
    from src.transformer_flux import FluxTransformer2DModel
    from src.lora_helper import set_single_lora, set_multi_lora
except Exception as e:

    raise e

# 用于缓存清理的函数
def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# 1. 基础模型加载节点
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
        
        # 初始化模型
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

# 新增节点：FLUX风格LoRA加载器
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
        # 获取LoRA文件的完整路径
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        # 加载LoRA权重
        print(f"加载FLUX风格LoRA: {lora_name}, 权重: {lora_weight}")
        
        # 如果weight_name为空，使用lora_name作为weight_name
        if not weight_name:
            weight_name = lora_name
            
        # 加载LoRA权重
        pipe.load_lora_weights(lora_path, weight_name=weight_name)
        
        # 融合LoRA
        pipe.fuse_lora(lora_weights=[lora_weight])
        
        # 确保模型在设备上
        device = get_torch_device()
        pipe.to(device)
        
        return (pipe,)

# 2. 控制模型选择节点
class EasyControlModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_type": (["canny", "depth", "hedsketch", "pose", "seg", "inpainting", "subject"], {"default": "canny"}),
                "transformer": ("EASY_CONTROL_TRANSFORMER",),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "cond_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "use_plugin_path": ("BOOLEAN", {"default": True, "label": "使用插件目录"}),
            }
        }
    
    RETURN_TYPES = ("EASY_CONTROL_TRANSFORMER",)
    FUNCTION = "select_control_model"
    CATEGORY = "EasyControl"

    def select_control_model(self, control_type, transformer, lora_weight, cond_size, use_plugin_path):
        # 选择模型目录
        if use_plugin_path:
            # 使用插件目录下的models文件夹
            lora_path = MODELS_PATH
        else:
            # 使用ComfyUI默认的models/easycontrol目录
            lora_path = os.path.join(folder_paths.models_dir, "easycontrol")
            os.makedirs(lora_path, exist_ok=True)
        
        # 控制模型路径映射
        control_models = {
            "canny": f"{lora_path}/canny.safetensors",
            "depth": f"{lora_path}/depth.safetensors",
            "hedsketch": f"{lora_path}/hedsketch.safetensors",
            "pose": f"{lora_path}/pose.safetensors",
            "seg": f"{lora_path}/seg.safetensors",
            "inpainting": f"{lora_path}/inpainting.safetensors",
            "subject": f"{lora_path}/subject.safetensors",
        }
        
        # 如果模型文件不存在，尝试从HuggingFace下载
        if not os.path.exists(control_models[control_type]):
            print(f"Downloading {control_type} model to {lora_path}...")
            hf_hub_download(
                repo_id="Xiaojiu-Z/EasyControl", 
                filename=f"models/{control_type}.safetensors", 
                local_dir=lora_path
            )
        
        # 加载控制模型
        set_single_lora(transformer, control_models[control_type], lora_weights=[lora_weight], cond_size=cond_size)
        
        return (transformer,)

# 3. 条件图像上传节点
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
        # 将ComfyUI格式转换为PIL格式
        image_np = 255. * image.cpu().numpy()
        image_pil = Image.fromarray(np.clip(image_np[0], 0, 255).astype(np.uint8))
        
        # 返回PIL图像以便后续处理
        return (image_pil,)

# 4. 文本编码节点
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
        # 这里仅传递文本，实际编码会在采样器中进行
        # 因为FLUX模型可能有自己的文本处理方式
        return (text,)

# 5. EasyControl采样器节点
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
        # 设置生成器种子
        generator = torch.Generator("cpu").manual_seed(seed)
        
        # 执行图像生成
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
        
        # 清理缓存
        clear_cache(transformer)
        
        # 获取生成的图像并转换为ComfyUI格式
        image = output.images[0]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return (image_tensor,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "EasyControlBaseModelLoader": EasyControlBaseModelLoader,
    "EasyControlLoraLoader": EasyControlLoraLoader,
    "EasyControlModelSelector": EasyControlModelSelector,
    "EasyControlSpatialImage": EasyControlSpatialImage,
    "EasyControlTextEncode": EasyControlTextEncode,
    "EasyControlSampler": EasyControlSampler,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyControlBaseModelLoader": "EasyControl 基础模型加载",
    "EasyControlLoraLoader": "EasyControl LoRA加载",
    "EasyControlModelSelector": "EasyControl 控制模型选择",
    "EasyControlSpatialImage": "EasyControl 条件图像准备",
    "EasyControlTextEncode": "EasyControl 文本编码",
    "EasyControlSampler": "EasyControl 采样器",
}