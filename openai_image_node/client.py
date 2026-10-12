"""One explicit POST per execution. No automatic retries of billable requests."""

import base64
import binascii
import io

import numpy as np
import requests
from PIL import Image, ImageOps, UnidentifiedImageError

API_URL = "https://api.openai.com/v1/images/generations"


def generate_images(api_key, payload, timeout):
    # The destination is fixed; an environment variable cannot redirect credentials.
    with requests.Session() as session:
        try:
            response = session.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=(15, timeout),
                allow_redirects=False,
            )
        except requests.Timeout:
            raise RuntimeError(
                "OpenAI image request timed out. It may already have been processed; "
                "check usage before running again."
            ) from None
        except requests.RequestException:
            raise RuntimeError("Could not connect to the OpenAI Images API.") from None
        with response:
            if response.status_code != 200:
                messages = {
                    400: "OpenAI rejected the image settings or prompt.",
                    401: "OpenAI rejected the API key.",
                    403: "The API key lacks model access; check project and organization access.",
                    404: "The selected model is unavailable for this project.",
                    429: "OpenAI rate or quota limit reached; check billing and rate limits.",
                }
                message = messages.get(response.status_code, "OpenAI image generation failed.")
                # Never echo the response body, headers, or requests exception (may contain keys).
                raise RuntimeError(f"{message} HTTP {response.status_code}.")
            try:
                body = response.json()
            except ValueError:
                raise RuntimeError("OpenAI returned an invalid JSON response.") from None
    if not isinstance(body, dict) or not isinstance(body.get("data"), list) or not body["data"]:
        raise RuntimeError("OpenAI returned no images.")
    if len(body["data"]) != payload.get("n", 1):
        raise RuntimeError("OpenAI returned an unexpected number of images.")
    return body


def decode_images(body):
    """Return lists of RGB [1,H,W,3] images and inverse-alpha [1,H,W] masks.

    Lists preserve original dimensions when auto sizing returns different image sizes.
    """
    import torch

    images, masks = [], []
    for item in body["data"]:
        encoded = item.get("b64_json") if isinstance(item, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError("OpenAI response is missing a base64 image.")
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
                pixels = np.array(rgba, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(pixels[:, :, :3].copy()).unsqueeze(0))
            masks.append(torch.from_numpy(1.0 - pixels[:, :, 3]).unsqueeze(0))
        except (ValueError, binascii.Error, UnidentifiedImageError, OSError):
            raise RuntimeError("OpenAI returned an invalid or unsupported encoded image.") from None
    return images, masks
