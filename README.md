# ComfyUI OpenAI Image Generation

Generate images with your own OpenAI API key, directly inside ComfyUI. The node uses
`POST https://api.openai.com/v1/images/generations` and returns standard ComfyUI images,
transparency masks, and usage metadata.

- An `api_key` input accepts a typed key or a connected `STRING` input.
- Blank `api_key` uses the server's `OPENAI_API_KEY` environment variable.
- Model, quality, background, image count, format, and compression controls.
- Standard or custom dimensions, with validation of documented model restrictions.
- A weekly GitHub Actions job updates the node's capability data and commits tested changes.

The initial default is `gpt-image-2.5-flare`. See the automatically generated
[supported API and model table](docs/supported-api.md) for the current catalog.
This integration implements text-to-image generation through the Images API.

## Install

In your ComfyUI directory:

```sh
git clone https://github.com/busyzeus/ComfyUI-OpenAI-Image-Generation.git custom_nodes/ComfyUI-OpenAI-Image-Generation
python -m pip install -r custom_nodes/ComfyUI-OpenAI-Image-Generation/requirements.txt
```

Use the **same Python environment as ComfyUI**, then restart ComfyUI. For the Windows
portable distribution, run this from the directory containing `python_embeded`:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-OpenAI-Image-Generation\requirements.txt
```

Python 3.10 or newer is required. PyTorch comes from your existing ComfyUI installation;
the requirements deliberately leave that installation in place.

## Use

1. Add **image → OpenAI → OpenAI Image Generation**.
2. Enter your API key and prompt, then choose a model and settings.
3. Connect `images` to **Save Image** or **Preview Image**, and queue the workflow.
4. Change `generation_id` to request another image with the same settings. Its control
   after generation can be set to increment or randomize; it is a cache control, not an
   API seed. Keeping it fixed lets ComfyUI reuse the previous result in its execution cache.

Open [examples/workflow.json](examples/workflow.json) in ComfyUI for a ready-to-wire
generation workflow. [examples/api-prompt.json](examples/api-prompt.json) is the equivalent
API-format graph to use in a `/prompt` request's `prompt` field. Both have a blank key.

`images` and `masks` are ComfyUI lists, with one item per generated image. This preserves
each image's dimensions when `size=auto` returns different sizes. Each image is an RGB
float tensor `[1, height, width, 3]`; each mask is `[1, height, width]`, where 1 means
transparent and 0 means opaque. To preserve alpha when saving, connect the image and
mask to **Join Image with Alpha**, then to **Save Image**.

`output_compression` is sent only for JPEG and WebP. Transparent output requires PNG
or WebP. `additional_parameters` accepts a JSON object for extra parameters present in
the checked-in API schema, such as `{"user":"your-user-id"}`. It cannot override the
named inputs or enable streaming. New optional API fields become usable there after sync.

The API key is a normal ComfyUI input, **not secret storage**. ComfyUI may put literal
inputs in saved workflows, queue history, and saved image metadata, including values
from connected string nodes. Use `OPENAI_API_KEY` with a blank node input when sharing
workflows or images. The node does not log keys or echo API error bodies.

OpenAI API billing and model access apply. A ChatGPT subscription does not provide an
API key. Requests use a fixed OpenAI endpoint and a configurable read timeout (default
300 seconds). Failed requests are not automatically retried; after a timeout, check
usage before re-running because the server may already have processed the request.

## Weekly automatic updates

[Update OpenAI API support](.github/workflows/update-api-support.yml) runs every Monday
at **00:23 UTC / 09:23 Asia/Seoul** and can also be run from GitHub's Actions tab.

The workflow:

1. Downloads OpenAI's official OpenAPI specification, image generation guide, and
   deprecation notices. No OpenAI key or paid API request is needed.
2. Refreshes model choices, supported option values, request validation, image limits,
   per-model quality/custom-size rules, retirement dates, and the API support report.
3. Runs lint and tests against the updated node.
4. Commits only changed generated files to the repository's default branch using
   `GITHUB_TOKEN`. An unchanged check creates no commit.

The node reads `openai_image_node/capabilities.json`, so these commits update its actual
behavior. This is a deterministic capability update, not unrestricted AI code rewriting.
New endpoints, incompatible required parameters, response changes, or unrecognized
documentation formats require an implementation change; the job fails before committing
instead of claiming support. The current working catalog remains usable. Runtime also
excludes models once their recorded retirement date arrives.

Actions must be enabled and repository rules must allow `github-actions[bot]` to push
to the default branch. The workflow requests `contents: write`; it does not bypass branch
protection. Scheduled jobs run from the default branch and GitHub may delay their start.
GitHub can disable schedules in inactive public repositories; re-enable the workflow
from Actions if needed. The job runs its own tests because commits made with
`GITHUB_TOKEN` do not normally trigger another push workflow.

Installed copies update with `git pull` in the custom node directory followed by a
ComfyUI restart. The GitHub schedule updates this repository, not every local installation.

To check or update manually:

```sh
python scripts/sync_capabilities.py --check
python scripts/sync_capabilities.py
python scripts/sync_capabilities.py --offline --check
```

The last command checks regeneration against the saved source snapshot. Retirement dates
are evaluated on the day it runs, so a snapshot can need regeneration after a shutdown.

## Development

```sh
python -m pip install -r requirements-dev.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ruff check .
ruff format --check .
python -m pytest
```

Tests use synthetic images and mocked HTTP requests; they never require credentials or
spend API credits. CI tests Python 3.10 and 3.13. The live sync command checks the actual
public API sources. A paid end-to-end generation test requires your own API key and access.

References: [OpenAI Images API](https://developers.openai.com/api/reference/resources/images/methods/generate),
[image generation guide](https://developers.openai.com/api/docs/guides/image-generation),
[deprecations](https://developers.openai.com/api/docs/deprecations),
[ComfyUI custom node interface](https://docs.comfy.org/custom-nodes/backend/server_overview).
