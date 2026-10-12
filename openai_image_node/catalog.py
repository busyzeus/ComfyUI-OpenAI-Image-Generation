"""Capability data is generated from OpenAI's public specification, never at startup."""

import json
import re
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator

CATALOG_PATH = Path(__file__).with_name("capabilities.json")
BLOCKED_PARAMETERS = {"response_format", "style", "stream", "partial_images"}


def load_catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def active_models(catalog, today=None):
    today = today or date.today()
    return [
        model
        for model in catalog["models"]
        if model not in catalog["retirements"]
        or date.fromisoformat(catalog["retirements"][model]) > today
    ]


def validate_request(payload, catalog):
    if payload.get("model") not in active_models(catalog):
        raise ValueError("Select an active image model from the node's model list.")
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        raise ValueError("Prompt must contain text.")
    for key in payload:
        if key in BLOCKED_PARAMETERS or key not in catalog["request_schema"]["properties"]:
            raise ValueError(f"Unsupported image parameter: {key}.")
    # Do not include validation exception text: it can echo user-provided data.
    error = next(Draft202012Validator(catalog["request_schema"]).iter_errors(payload), None)
    if error:
        field = str(next(iter(error.path), "request"))
        raise ValueError(f"Invalid {field}; check the supported API values and limits.")
    model = payload["model"]
    if payload.get("quality", "auto") not in catalog["model_qualities"][model]:
        raise ValueError(f"The selected quality is not supported by {model}.")
    size = payload.get("size", "auto")
    if size not in catalog["sizes"]:
        if model not in catalog["custom_size_models"]:
            raise ValueError("This model requires a standard size or auto.")
        match = re.fullmatch(r"([1-9][0-9]{0,4})x([1-9][0-9]{0,4})", size)
        if not match:
            raise ValueError("Size must be auto or WIDTHxHEIGHT, for example 1536x864.")
        width, height = map(int, match.groups())
        limits = catalog["size_limits"]
        if (
            width % limits["multiple"]
            or height % limits["multiple"]
            or max(width, height) > limits["max_edge"]
            or max(width, height) > min(width, height) * limits["max_ratio"]
            or not limits["min_pixels"] <= width * height <= limits["max_pixels"]
        ):
            raise ValueError("Custom size exceeds the documented edge, ratio, or pixel limits.")
    if payload.get("background") == "transparent" and payload.get("output_format", "png") == "jpeg":
        raise ValueError("Transparent backgrounds require png or webp output.")
    if "output_compression" in payload:
        compression = payload["output_compression"]
        if type(compression) is not int or not 0 <= compression <= 100:
            raise ValueError("Output compression must be an integer between 0 and 100.")
    if len(payload["prompt"]) > catalog["prompt_max_length"]:
        raise ValueError("Prompt exceeds the documented maximum length.")
