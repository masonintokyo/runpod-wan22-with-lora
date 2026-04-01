# Wan 2.2 RunPod Serverless Worker
[한국어 README 보기](README_kr.md)

This repository contains a `runpod.io` Serverless worker and a Python client for Wan 2.2 image-to-video generation using ComfyUI.

## What Changed

- Added selectable Wan 2.2 model profiles
- Added GGUF quantized Wan 2.2 support
- Added automatic selection from `gpu_profile` or `target_vram_gb`
- Added RunPod-friendly output modes: `base64` or `bucket_url`
- Added lazy download for Wan 2.2 experts, support assets, and default LoRAs into `/runpod-volume`

## Main Files

- `handler.py`: RunPod handler, model selection, lazy downloads, workflow execution
- `generate_video_client.py`: Python client for submitting jobs
- `entrypoint.sh`: Starts ComfyUI and the handler
- `new_Wan22_api.json`: Single-image workflow
- `new_Wan22_flf2v_api.json`: Start/end image workflow
- `AGENTS.md`: Maintenance notes for agents and developers
- `USER_GUIDE.md`: Japanese usage guide

## Supported Model Profiles

- `fp8_e4m3fn`
- `fp8_e5m2`
- `gguf_q2_k`
- `gguf_q3_k_m`
- `gguf_q4_k_m`
- `gguf_q5_k_m`
- `gguf_q6_k`
- `gguf_q8_0`

Recommended starting points:

- 24GB class GPUs: `gguf_q4_k_m`
- 48GB+ class GPUs: `fp8_e4m3fn`

## Request Fields

Core fields:

- `prompt`
- one of `image_path`, `image_url`, `image_base64`
- optional `end_image_path`, `end_image_url`, `end_image_base64`
- `width`, `height`, `length`, `steps`, `seed`, `cfg`
- optional `lora_pairs`

New selection fields:

- `model_profile`
- `gpu_profile`
- `target_vram_gb`
- `output_mode`
- `refresh_worker`

## Python Client Example

```python
from generate_video_client import GenerateVideoClient

client = GenerateVideoClient(
    runpod_endpoint_id="your-endpoint-id",
    runpod_api_key="your-runpod-api-key",
)

result = client.create_video_from_image(
    image_path="./example_image.png",
    prompt="cinematic portrait, subtle motion, realistic lighting",
    width=480,
    height=832,
    length=81,
    steps=10,
    seed=42,
    cfg=2.0,
    model_profile="gguf_q4_k_m",
    output_mode="auto",
)

if result.get("status") == "COMPLETED":
    client.save_video_result(result, "./output_video.mp4")
```

## Example Request

```json
{
  "input": {
    "prompt": "cinematic portrait of a woman turning toward the camera",
    "image_url": "https://example.com/input.png",
    "width": 480,
    "height": 832,
    "length": 81,
    "steps": 10,
    "seed": 42,
    "cfg": 2.0,
    "model_profile": "gguf_q4_k_m",
    "output_mode": "auto"
  }
}
```

## Deployment Flow

1. Fork or push this repository to your GitHub account.
2. In RunPod Serverless, create a new endpoint from that repository or from a built image.
3. Mount a network volume to `/runpod-volume` if you want model and LoRA persistence.
4. Deploy the endpoint.
5. Send the first request with an input image.

The first request is the one most likely to trigger model download and cold start.
The Docker image no longer bakes Wan weights into the image, which keeps deployment lighter and moves storage to the network volume or local worker cache.

## First Request Recommendations

- If your endpoint uses a 24GB GPU class, send `model_profile: "gguf_q4_k_m"`.
- If your endpoint uses a 48GB or 80GB GPU class, send `model_profile: "fp8_e4m3fn"`.
- Start without custom LoRAs first.
- Use a prompt that describes motion, not just the subject.

Example prompt:

`cinematic portrait of a woman slowly turning toward the camera, gentle blinking, subtle hair motion, natural lighting`

## LoRA Usage

Default workflow LoRAs:

- `high_noise_model.safetensors`
- `low_noise_model.safetensors`

These are fetched lazily on first use and then reused from the volume or worker-local model directory.

Custom LoRAs:

- Put them in `/runpod-volume/loras`
- Pass them as `lora_pairs`
- Start with `1.0`
- If results become unstable, reduce to `0.7` to `0.9`

## cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "cinematic portrait of a woman slowly turning toward the camera, gentle blinking, subtle hair motion, natural lighting",
      "image_url": "https://example.com/input.png",
      "width": 480,
      "height": 832,
      "length": 81,
      "steps": 10,
      "seed": 42,
      "cfg": 2.0,
      "model_profile": "gguf_q4_k_m",
      "output_mode": "auto"
    }
  }'
```

## Local UI

This repository includes a local web UI in [`local_ui_server.py`](/Users/jimmy/Projects/generate_video/local_ui_server.py).

Run it like this:

```bash
python -m pip install flask requests
export RUNPOD_ENDPOINT_ID=your-endpoint-id
export RUNPOD_API_KEY=your-api-key
python local_ui_server.py
```

Then open `http://127.0.0.1:8787`.

## RunPod Notes

- Physical GPU choice is configured on the RunPod endpoint, not inside the handler.
- `gpu_profile` is an optimization target for model selection, not infrastructure provisioning.
- Use `/runpod-volume/models` and `/runpod-volume/loras` for better cold-start behavior.
- The current Docker image is intentionally thinner: ComfyUI + custom nodes are baked in, while heavy Wan assets are downloaded only when needed.
- Prefer `output_mode="bucket_url"` when bucket credentials are configured.

## References

- RunPod handler docs: https://docs.runpod.io/serverless/workers/handlers/handler-generator
- RunPod environment variables and bucket uploads: https://docs.runpod.io/serverless/development/environment-variables
- RunPod endpoint configuration and payload limits: https://docs.runpod.io/serverless/endpoints/endpoint-configurations
- Wan 2.2 model card: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- ComfyUI WanVideo wrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper
