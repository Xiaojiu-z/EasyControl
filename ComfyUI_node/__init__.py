from .nodes import (
    EasyControlBaseModelLoader,
    EasyControlLoraLoader,
    EasyControlModelSelector,
    EasyControlSpatialImage,
    EasyControlTextEncode,
    EasyControlSampler
)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "EasyControlBaseModelLoader": EasyControlBaseModelLoader,
    "EasyControlLoraLoader": EasyControlLoraLoader,
    "EasyControlModelSelector": EasyControlModelSelector,
    "EasyControlSpatialImage": EasyControlSpatialImage,
    "EasyControlTextEncode": EasyControlTextEncode,
    "EasyControlSampler": EasyControlSampler
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyControlBaseModelLoader": "EasyControl Base Model Loader",
    "EasyControlLoraLoader": "EasyControl LoRA Loader",
    "EasyControlModelSelector": "EasyControl Control Model Selector",
    "EasyControlSpatialImage": "EasyControl Condition Image Preparation",
    "EasyControlTextEncode": "EasyControl Text Encoder",
    "EasyControlSampler": "EasyControl Image Generator"
}

# Export these mappings for ComfyUI use
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("EasyControl_new nodes loaded") 