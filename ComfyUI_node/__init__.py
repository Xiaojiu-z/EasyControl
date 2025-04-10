from .nodes import (
    EasyControlBaseModelLoader,
    EasyControlLoraLoader,
    EasyControlModelSelector,
    EasyControlSpatialImage,
    EasyControlTextEncode,
    EasyControlSampler
)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "EasyControlBaseModelLoader": EasyControlBaseModelLoader,
    "EasyControlLoraLoader": EasyControlLoraLoader,
    "EasyControlModelSelector": EasyControlModelSelector,
    "EasyControlSpatialImage": EasyControlSpatialImage,
    "EasyControlTextEncode": EasyControlTextEncode,
    "EasyControlSampler": EasyControlSampler
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyControlBaseModelLoader": "EasyControl 基础模型加载",
    "EasyControlLoraLoader": "EasyControl LoRA加载",
    "EasyControlModelSelector": "EasyControl 控制模型选择",
    "EasyControlSpatialImage": "EasyControl 条件图像准备",
    "EasyControlTextEncode": "EasyControl 文本编码",
    "EasyControlSampler": "EasyControl 图像生成"
}

# 导出这些映射供ComfyUI使用
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("EasyControl_new节点已加载") 