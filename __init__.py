"""ComfyUI discovers these mappings when this directory is in custom_nodes."""

from .openai_image_node.nodes import OpenAIImageGeneration

NODE_CLASS_MAPPINGS = {"OpenAIImageGeneration": OpenAIImageGeneration}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAIImageGeneration": "OpenAI Image Generation"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
