"""Refresh the executable capability catalog from official, public OpenAI sources.

No API key, language model, or generated Python code is involved. Unexpected source
structures fail before writes, leaving the working catalog intact.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "openapi": "https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.json",
    "guide": "https://developers.openai.com/api/docs/guides/image-generation.md",
    "deprecations": "https://developers.openai.com/api/docs/deprecations.md",
}
CORE_FIELDS = {
    "model",
    "prompt",
    "n",
    "size",
    "quality",
    "background",
    "output_format",
    "output_compression",
    "moderation",
}


def resolve(value, spec, trail=()):
    if isinstance(value, list):
        return [resolve(item, spec, trail) for item in value]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        ref = value["$ref"]
        if not ref.startswith("#/") or ref in trail:
            raise ValueError("Unexpected external or recursive API schema reference.")
        target = spec
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return resolve(
            {**target, **{k: v for k, v in value.items() if k != "$ref"}}, spec, (*trail, ref)
        )
    return {key: resolve(item, spec, trail) for key, item in value.items()}


def enum_values(schema):
    if schema.get("deprecated"):
        return []
    deprecated = schema.get("x-openai-enum-deprecations", {})
    values = [v for v in schema.get("enum", []) if isinstance(v, str) and v not in deprecated]
    for union in ("anyOf", "oneOf", "allOf"):
        for item in schema.get(union, []):
            values.extend(enum_values(item))
    return list(dict.fromkeys(values))


def parse_retirements(markdown, models):
    if "Shutdown date" not in markdown:
        raise ValueError("Deprecation table format changed.")
    result = {}
    in_table = False
    for line in markdown.splitlines():
        if not line.strip().startswith("|"):
            in_table = False
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) < 2:
            continue
        if cells[0] == "Shutdown date":
            in_table = True
            continue
        if not in_table:
            continue
        affected = set(re.findall(r"`([^`]+)`", cells[1])).intersection(models)
        if not affected:
            continue
        parsed = None
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                parsed = datetime.strptime(cells[0], fmt).date().isoformat()
                break
            except ValueError:
                pass
        if parsed is None:
            raise ValueError("An image model shutdown date needs manual review.")
        result.update(dict.fromkeys(sorted(affected), parsed))
    return dict(sorted(result.items()))


def number(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Documented image limits changed format; review the sync parser.")
    return int(match[1].replace(",", ""))


def extract_snapshot(spec, guide, deprecations):
    operation = spec["paths"]["/images/generations"]["post"]
    schema = resolve(operation["requestBody"]["content"]["application/json"]["schema"], spec)
    if not CORE_FIELDS.issubset(schema["properties"]):
        raise ValueError("A core image parameter was removed; node changes are required.")
    if set(schema.get("required", [])) != {"model", "prompt"}:
        raise ValueError("Required image parameters changed; node changes are required.")
    if any(schema["properties"][name].get("deprecated") for name in CORE_FIELDS):
        raise ValueError("A core image parameter was deprecated; node changes are required.")
    response = resolve(operation["responses"]["200"]["content"]["application/json"]["schema"], spec)
    item = response["properties"]["data"]["items"]
    if "b64_json" not in item["properties"]:
        raise ValueError("The image response format changed; decoder changes are required.")
    models = enum_values(schema["properties"]["model"])
    if not models:
        raise ValueError("No documented image generation models found.")
    limits = {
        "multiple": number(guide, r"Width and height must be multiples of (\d+)"),
        "max_ratio": number(guide, r"aspect ratio must be between 1:\d+ and (\d+):1"),
        "max_edge": number(guide, r"neither edge may exceed (\d+) pixels"),
        "min_pixels": number(guide, r"total pixel count must be between ([\d,]+)"),
        "max_pixels": number(guide, r"total pixel count must be between [\d,]+ and ([\d,]+)"),
    }
    return {
        "sources": SOURCES,
        "request_schema": schema,
        "size_limits": limits,
        "retirements": parse_retirements(deprecations, models),
        "image_endpoints": sorted(path for path in spec["paths"] if path.startswith("/images/")),
    }


def build_catalog(snapshot, today=None, previous_default=None):
    today = today or date.today()
    schema = snapshot["request_schema"]
    fields = schema["properties"]
    models = [
        model
        for model in enum_values(fields["model"])
        if snapshot["retirements"].get(model, "9999-12-31") > today.isoformat()
    ]
    if not models:
        raise ValueError("Refusing to write an empty model catalog.")
    enums = {
        name: enum_values(fields[name])
        for name in ("quality", "background", "output_format", "moderation")
    }
    for name, values in enums.items():
        expected_default = "png" if name == "output_format" else "auto"
        if expected_default not in values:
            raise ValueError(f"Default {name} changed; review the node before updating.")
    # The spec describes per-model quality restrictions in its bullet list.
    base_qualities, restricted = set(), {}
    for bullet in re.split(r"\n\s*-\s+", fields["quality"]["description"]):
        tokens = set(re.findall(r"`([^`]+)`", bullet))
        qualities = tokens.intersection(enums["quality"])
        named_models = tokens.intersection(enum_values(fields["model"]))
        if named_models:
            for model in models:
                if any(model == name or model.startswith(name + "-") for name in named_models):
                    restricted.setdefault(model, set()).update(qualities)
        else:
            base_qualities.update(qualities)
    described = base_qualities | set().union(*restricted.values())
    if described != set(enums["quality"]) or "auto" not in base_qualities:
        raise ValueError("Model quality rules changed format; manual review is required.")
    model_qualities = {
        model: [
            value
            for value in enums["quality"]
            if value in base_qualities | restricted.get(model, set())
        ]
        for model in models
    }
    custom = re.search(
        r"For (.+?), arbitrary resolutions are supported", fields["size"]["description"], re.DOTALL
    )
    if not custom:
        raise ValueError("Custom size model rules changed format; manual review is required.")
    custom_models = set(re.findall(r"`([^`]+)`", custom[1]))
    minimum, maximum = fields["n"]["minimum"], fields["n"]["maximum"]
    if not (type(minimum) is int and type(maximum) is int and 1 <= minimum <= maximum):
        raise ValueError("Unexpected image count limits.")
    default = previous_default if previous_default in models else fields["model"].get("example")
    if default not in models:
        default = models[0]
    return {
        "schema_version": 1,
        "endpoint": "/images/generations",
        "default_model": default,
        "models": models,
        "enums": enums,
        "model_qualities": model_qualities,
        "sizes": enum_values(fields["size"]),
        "custom_size_models": [model for model in models if model in custom_models],
        "size_limits": snapshot["size_limits"],
        "n_limits": {"min": minimum, "max": maximum},
        "prompt_max_length": number(fields["prompt"]["description"], r"maximum length is (\d+)"),
        "retirements": snapshot["retirements"],
        "request_schema": {**schema, "additionalProperties": False},
    }


def render_report(catalog, snapshot):
    lines = [
        "# Supported OpenAI image API",
        "",
        "Generated by `scripts/sync_capabilities.py`; do not edit by hand.",
        "",
        "The node calls `POST /v1/images/generations`. Other image endpoints are recorded",
        "below for discovery; this node implements text-to-image generation only.",
        "",
        f"Default model: `{catalog['default_model']}`.",
        "",
        "| Model | Quality | Custom dimensions | Scheduled retirement |",
        "| --- | --- | --- | --- |",
    ]
    for model in catalog["models"]:
        qualities = ", ".join(catalog["model_qualities"][model])
        custom = "yes" if model in catalog["custom_size_models"] else "no"
        retirement = catalog["retirements"].get(model, "—")
        lines.append(f"| `{model}` | {qualities} | {custom} | {retirement} |")
    lines += [
        "",
        "Standard sizes: " + ", ".join(f"`{s}`" for s in catalog["sizes"]) + ".",
        "",
        "Custom size limits: `" + json.dumps(catalog["size_limits"], sort_keys=True) + "`.",
        "",
        "Documented image endpoints: "
        + ", ".join(f"`{path}`" for path in snapshot["image_endpoints"])
        + ".",
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- [{name}]({url})" for name, url in SOURCES.items())
    digest = hashlib.sha256(json_text(snapshot).encode()).hexdigest()
    lines += [
        "",
        f"Relevant source snapshot SHA-256: `{digest}`.",
        "",
        "Availability still depends on your OpenAI project. Models disappear from the",
        "node's menu on their documented shutdown date, even before the next weekly sync.",
        "",
    ]
    return "\n".join(lines)


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-OpenAI-API-Sync/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the checked-in source snapshot")
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if generated files would change"
    )
    args = parser.parse_args()
    snapshot_path = ROOT / "docs/api_snapshot.json"
    if args.offline:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        spec = json.loads(fetch(SOURCES["openapi"]))
        snapshot = extract_snapshot(spec, fetch(SOURCES["guide"]), fetch(SOURCES["deprecations"]))
    catalog_path = ROOT / "openai_image_node/capabilities.json"
    previous = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    catalog = build_catalog(snapshot, previous_default=previous.get("default_model"))
    # Finish all parsing and rendering before touching the last working catalog.
    outputs = {
        snapshot_path: json_text(snapshot),
        catalog_path: json_text(catalog),
        ROOT / "docs/supported-api.md": render_report(catalog, snapshot),
    }
    changed = [
        path
        for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if args.check:
        if changed:
            print("API capabilities need updating: " + ", ".join(p.name for p in changed))
            return 1
    else:
        for path in changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(outputs[path], encoding="utf-8", newline="\n")
    print(
        f"Checked official image API: {len(catalog['models'])} models, "
        f"{len(changed)} changed files."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, TypeError, ValueError) as error:
        sys.exit(f"API sync stopped without updating files: {error}")
