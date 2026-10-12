import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image

from openai_image_node import client, nodes
from openai_image_node.catalog import load_catalog, validate_request

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-test-only-never-a-real-key"


def image_data(size=(4, 3), alpha=128, fmt="PNG"):
    buffer = io.BytesIO()
    image = Image.new("RGBA", size, (255, 32, 0, alpha))
    if fmt == "JPEG":
        image = image.convert("RGB")
    image.save(buffer, format=fmt)
    return {"b64_json": base64.b64encode(buffer.getvalue()).decode()}


@pytest.fixture
def capture_request(monkeypatch):
    captured = {}

    def fake_generate(key, payload, timeout):
        captured.update(key=key, payload=payload, timeout=timeout)
        return {"data": [image_data() for _ in range(payload["n"])], "usage": {"total_tokens": 42}}

    monkeypatch.setattr(nodes, "generate_images", fake_generate)
    return captured


def run_node(**kwargs):
    inputs = {"api_key": KEY, "prompt": "A red kite", "model": load_catalog()["default_model"]}
    return nodes.OpenAIImageGeneration().generate(**(inputs | kwargs))


def test_comfy_package_registration():
    spec = importlib.util.spec_from_file_location(
        "custom_openai_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        node = module.NODE_CLASS_MAPPINGS["OpenAIImageGeneration"]
        assert node.RETURN_TYPES == ("IMAGE", "MASK", "STRING")
        assert node.INPUT_TYPES()["required"]["api_key"][0] == "STRING"
    finally:
        for name in list(sys.modules):
            if name.startswith("custom_openai_test"):
                del sys.modules[name]


def test_generation_input_output_contract(capture_request):
    images, masks, metadata = run_node(n=2, generation_id=4321)
    assert len(images) == len(masks) == 2
    assert images[0].shape == (1, 3, 4, 3)
    assert masks[0].shape == (1, 3, 4)
    assert images[0].dtype == torch.float32
    assert images[0][0, 0, 0, 0] == 1
    assert masks[0][0, 0, 0].item() == pytest.approx(1 - 128 / 255)
    assert "output_compression" not in capture_request["payload"]
    assert "generation_id" not in capture_request["payload"]
    assert "api_key" not in capture_request["payload"]
    assert capture_request["key"] == KEY
    assert json.loads(metadata)["usage"]["total_tokens"] == 42
    assert KEY not in metadata


def test_key_fallback_and_explicit_precedence(capture_request, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-test")
    run_node(api_key="  ")
    assert capture_request["key"] == "sk-environment-test"
    run_node(api_key="  " + KEY + "  ")
    assert capture_request["key"] == KEY


def test_empty_key_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        run_node(api_key="")


@pytest.mark.parametrize("fmt", ["jpeg", "webp"])
def test_compression_sent_for_supported_formats(capture_request, fmt):
    run_node(output_format=fmt, output_compression=80)
    assert capture_request["payload"]["output_compression"] == 80


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": " "},
        {"prompt": "x" * (load_catalog()["prompt_max_length"] + 1)},
        {"n": 0},
        {"n": load_catalog()["n_limits"]["max"] + 1},
        {"model": "dall-e-3"},
        {"quality": "hd"},
        {"size": "1x1"},
        {"size": "1025x1024"},
        {"size": "3840x3840"},
        {"size": "4096x1024"},
        {"size": "3200x800"},
        {"background": "transparent", "output_format": "jpeg"},
        {"output_format": "jpeg", "output_compression": 101},
        {"timeout_seconds": 0},
        {"additional_parameters": "[]"},
        {"additional_parameters": "invalid"},
        {"additional_parameters": '{"model":"other"}'},
        {"additional_parameters": '{"stream":true}'},
        {"additional_parameters": '{"unexpected":123}'},
        {"api_key": "sk-test\ninvalid"},
        {"api_key": "sk-test\x00invalid"},
        {"api_key": "sk-test-\u2603"},
    ],
)
def test_invalid_inputs_never_call_api(capture_request, kwargs):
    with pytest.raises(ValueError):
        run_node(**kwargs)
    assert not capture_request


def test_additional_documented_parameter(capture_request):
    run_node(additional_parameters='{"user":"test-user"}')
    assert capture_request["payload"]["user"] == "test-user"


def test_custom_dimensions_and_extended_quality(capture_request):
    run_node(size="1536x864", quality="max")
    assert capture_request["payload"]["size"] == "1536x864"


def test_old_model_rejects_new_quality(capture_request):
    with pytest.raises(ValueError, match="quality"):
        run_node(model="gpt-image-2", quality="max")
    assert not capture_request


def test_decodes_mixed_dimensions_and_formats():
    images, masks = client.decode_images(
        {
            "data": [
                image_data((4, 3)),
                image_data((2, 5), fmt="JPEG"),
                image_data((3, 2), fmt="WEBP"),
            ]
        }
    )
    assert [tuple(image.shape) for image in images] == [(1, 3, 4, 3), (1, 5, 2, 3), (1, 2, 3, 3)]
    assert torch.count_nonzero(masks[1]) == 0


@pytest.mark.parametrize("item", [{}, {"b64_json": "bad!"}, {"b64_json": "dGVzdA=="}, None])
def test_bad_image_data(item):
    with pytest.raises(RuntimeError):
        client.decode_images({"data": [item]})


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def json(self):
        return self.body


def test_http_request_auth_timeout_and_destination(monkeypatch):
    calls = []

    def post(self, url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(body={"data": [image_data()]})

    monkeypatch.setattr(requests.Session, "post", post)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    client.generate_images(KEY, {"prompt": "test"}, 123)
    assert len(calls) == 1
    url, options = calls[0]
    assert url == "https://api.openai.com/v1/images/generations"
    assert options["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert options["timeout"] == (15, 123)
    assert options["allow_redirects"] is False


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 429, 500])
def test_api_errors_do_not_echo_secrets_or_retry(monkeypatch, status):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(status, {"error": {"message": KEY}})

    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(RuntimeError) as error:
        client.generate_images(KEY, {}, 300)
    assert str(status) in str(error.value)
    assert KEY not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("exception", [requests.Timeout(KEY), requests.ConnectionError(KEY)])
def test_transport_errors_do_not_echo_secrets(monkeypatch, exception):
    def post(*args, **kwargs):
        raise exception

    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(RuntimeError) as error:
        client.generate_images(KEY, {}, 300)
    assert KEY not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("body", [None, [], {}, {"data": []}, {"data": [image_data()]}])
def test_missing_or_incomplete_response(monkeypatch, body):
    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: FakeResponse(body=body))
    with pytest.raises(RuntimeError):
        client.generate_images(KEY, {"n": 2}, 300)


def test_every_catalog_model_has_valid_basic_settings():
    catalog = load_catalog()
    from openai_image_node.catalog import active_models

    for model in active_models(catalog):
        validate_request({"model": model, "prompt": "test", "size": "1024x1024"}, catalog)
