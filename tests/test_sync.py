import copy
import json
from datetime import date
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openai_image_node.catalog import active_models, load_catalog
from scripts.sync_capabilities import (
    build_catalog,
    enum_values,
    extract_snapshot,
    json_text,
    parse_retirements,
    resolve,
)

ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 12)


@pytest.fixture
def snapshot():
    return json.loads((ROOT / "docs/api_snapshot.json").read_text(encoding="utf-8"))


def test_catalog_schema_and_generation_are_deterministic(snapshot):
    catalog = build_catalog(snapshot, today=AS_OF)
    assert json_text(catalog) == json_text(build_catalog(copy.deepcopy(snapshot), today=AS_OF))
    Draft202012Validator.check_schema(catalog["request_schema"])
    assert "dall-e-3" not in catalog["models"]
    assert "hd" not in catalog["enums"]["quality"]
    assert catalog["default_model"] in catalog["models"]


def test_new_models_options_and_parameters_update_runtime(snapshot, monkeypatch):
    fields = snapshot["request_schema"]["properties"]
    fields["model"]["anyOf"][1]["enum"].append("gpt-image-future")
    fields["moderation"]["anyOf"][0]["enum"].append("future-setting")
    fields["future_parameter"] = {"type": "integer", "minimum": 1}
    catalog = build_catalog(snapshot, today=AS_OF)
    assert "gpt-image-future" in catalog["models"]
    assert "future-setting" in catalog["enums"]["moderation"]
    assert "future_parameter" in catalog["request_schema"]["properties"]
    from openai_image_node import nodes

    monkeypatch.setattr(nodes, "load_catalog", lambda: catalog)
    inputs = nodes.OpenAIImageGeneration.INPUT_TYPES()["required"]
    assert "gpt-image-future" in inputs["model"][0]
    assert "future-setting" in inputs["moderation"][0]


def test_retirements_removed_by_sync_and_runtime(snapshot):
    retiring_model = "test-retiring-image-model"
    snapshot["request_schema"]["properties"]["model"]["anyOf"][1]["enum"].append(retiring_model)
    snapshot["retirements"][retiring_model] = "2026-10-23"
    before = build_catalog(snapshot, today=AS_OF, previous_default=retiring_model)
    assert retiring_model in active_models(before, date(2026, 10, 22))
    assert retiring_model not in active_models(before, date(2026, 10, 23))
    after = build_catalog(snapshot, today=date(2026, 12, 1), previous_default=retiring_model)
    assert not set(snapshot["retirements"]).intersection(after["models"])
    assert after["default_model"] in after["models"]


def test_retirement_date_formats_and_replacements():
    document = """
| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| October 23, 2026 | `old-one` | `new-one` |
| Dec 1, 2026 | `old-two` | `new-one` |
| 2026-05-12 | `old-three` | `new-one` |
"""
    dates = parse_retirements(document, ["old-one", "old-two", "old-three", "new-one"])
    assert dates == {"old-one": "2026-10-23", "old-two": "2026-12-01", "old-three": "2026-05-12"}


def test_bad_dates_fail_closed():
    with pytest.raises(ValueError):
        parse_retirements("| Shutdown date | Model |\n| TBD | `test-model` |", ["test-model"])


def test_schema_references():
    spec = {"components": {"schemas": {"X": {"type": "string", "enum": ["yes"]}}}}
    assert enum_values(resolve({"$ref": "#/components/schemas/X"}, spec)) == ["yes"]
    with pytest.raises(ValueError):
        resolve({"$ref": "https://example.invalid/schema"}, spec)
    spec["components"]["schemas"]["X"] = {"$ref": "#/components/schemas/X"}
    with pytest.raises(ValueError):
        resolve(spec["components"]["schemas"]["X"], spec)


def test_unknown_quality_rule_requires_review(snapshot):
    snapshot["request_schema"]["properties"]["quality"]["anyOf"][0]["enum"].append("mystery")
    with pytest.raises(ValueError, match="quality rules"):
        build_catalog(snapshot, today=AS_OF)


def test_breaking_request_contract_requires_review(snapshot):
    schema = snapshot["request_schema"]
    schema["required"].append("new_required_field")
    spec = {
        "paths": {
            "/images/generations": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": schema}}},
                }
            }
        }
    }
    with pytest.raises(ValueError, match="Required image parameters"):
        extract_snapshot(spec, "", "")


def test_generated_manifest_is_usable():
    catalog = load_catalog()
    assert catalog["endpoint"] == "/images/generations"
    assert set(catalog["models"]) == set(catalog["model_qualities"])
    assert all(catalog["enums"].values())
