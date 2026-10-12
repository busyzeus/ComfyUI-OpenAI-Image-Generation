import json
import os

from .catalog import active_models, load_catalog, validate_request
from .client import decode_images, generate_images


class OpenAIImageGeneration:
    CATEGORY = "image/OpenAI"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "metadata")
    OUTPUT_IS_LIST = (True, True, False)
    DESCRIPTION = (
        "Generate images with your OpenAI API key. Change generation_id for a new request."
    )

    @classmethod
    def INPUT_TYPES(cls):
        catalog = load_catalog()
        models = active_models(catalog)
        default = catalog["default_model"]
        if default not in models and models:
            default = models[0]
        enums = catalog["enums"]
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Your OpenAI key. Blank uses OPENAI_API_KEY. "
                            "Literal inputs may be saved in workflows."
                        ),
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (models, {"default": default}),
                "size": ("STRING", {"default": "1024x1024"}),
                "quality": (enums["quality"], {"default": "auto"}),
                "n": ("INT", {"default": 1, **catalog["n_limits"]}),
                "background": (enums["background"], {"default": "auto"}),
                "output_format": (enums["output_format"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "moderation": (enums["moderation"], {"default": "auto"}),
                "generation_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9007199254740991,
                        "control_after_generate": True,
                        "tooltip": (
                            "Change to request a new image. "
                            "This is a cache control, not an OpenAI seed."
                        ),
                    },
                ),
            },
            "optional": {
                "timeout_seconds": ("INT", {"default": 300, "min": 30, "max": 1800}),
                "additional_parameters": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": (
                            "JSON for additional parameters in the synced API schema. "
                            "Cannot override node inputs."
                        ),
                    },
                ),
            },
        }

    def generate(
        self,
        api_key,
        prompt,
        model,
        size="1024x1024",
        quality="auto",
        n=1,
        background="auto",
        output_format="png",
        output_compression=100,
        moderation="auto",
        generation_id=0,
        timeout_seconds=300,
        additional_parameters="{}",
    ):
        key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not key or any(not 33 <= ord(character) <= 126 for character in key):
            raise ValueError("Supply an OpenAI API key in api_key or OPENAI_API_KEY.")
        if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 1800:
            raise ValueError("Timeout must be an integer between 30 and 1800 seconds.")
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
            "background": background,
            "output_format": output_format,
            "moderation": moderation,
        }
        if output_format in ("jpeg", "webp"):
            payload["output_compression"] = output_compression
        try:
            extras = json.loads(additional_parameters)
        except (TypeError, ValueError):
            raise ValueError("Additional parameters must be a JSON object.") from None
        if not isinstance(extras, dict):
            raise ValueError("Additional parameters must be a JSON object.")
        reserved = set(payload) | {
            "output_compression",
            "api_key",
            "generation_id",
            "timeout_seconds",
        }
        if reserved.intersection(extras):
            raise ValueError("Additional parameters cannot override node inputs.")
        payload.update(extras)
        validate_request(payload, load_catalog())
        body = generate_images(key, payload, timeout_seconds)
        images, masks = decode_images(body)
        metadata = {name: body[name] for name in ("created", "usage") if name in body}
        metadata.update(model=model, image_count=len(images))
        return images, masks, json.dumps(metadata, ensure_ascii=False)
