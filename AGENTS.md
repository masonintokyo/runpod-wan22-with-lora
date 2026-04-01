# AGENTS Guide

## Purpose

This repository provides a `runpod.io` Serverless worker for Wan 2.2 image-to-video generation on top of ComfyUI.

The worker supports:

- Wan 2.2 FP8 model selection
- Wan 2.2 GGUF quantized model selection
- VRAM-driven automatic profile selection
- GPU profile driven automatic profile selection
- RunPod-friendly output modes (`base64` or bucket upload URL)
- Thin Docker image strategy: ComfyUI/custom nodes in image, heavy Wan assets lazy-downloaded

## Repository Structure

- `handler.py`
  RunPod Serverless entrypoint. Handles input validation, model profile resolution, lazy model download, ComfyUI prompt submission, and result packaging.
- `generate_video_client.py`
  Reference Python client for `/run` style RunPod jobs.
- `entrypoint.sh`
  Starts ComfyUI and then the RunPod handler.
- `new_Wan22_api.json`
  Single-image Wan 2.2 workflow.
- `new_Wan22_flf2v_api.json`
  Start/end image workflow.
- `extra_model_paths.yaml`
  Includes `/runpod-volume/models` and `/runpod-volume/loras`, which is important for RunPod deployment efficiency.
- `pyproject.toml`
  Python dependency management for local development via `uv`. Covers the local UI, reference client, and handler-side Python packages used in this repository.

## Local Python Dependency Management

- Use `uv sync` for local environment setup.
- Use `uv run python local_ui_server.py` to start the local UI.
- When adding Python dependencies for local scripts, update `pyproject.toml` instead of documenting ad-hoc `pip install ...` commands.

## Current Model Selection Design

`handler.py` owns model selection. The workflow JSON stays generic and the handler injects the actual HIGH/LOW expert filenames into node `122` and node `549`.

Supported request fields:

- `model_profile`
  Explicit profile such as `fp8_e4m3fn`, `fp8_e5m2`, `gguf_q4_k_m`, `gguf_q5_k_m`.
- `gpu_profile`
  Optimization target such as `L4_24GB`, `RTX_4090_24GB`, `L40S_48GB`, `A100_80GB`.
- `target_vram_gb`
  Automatic profile selection by VRAM target.

Selection precedence:

1. `model_profile`
2. `gpu_profile`
3. `target_vram_gb`
4. `DEFAULT_MODEL_PROFILE` env var
5. fallback `fp8_e4m3fn`

## RunPod-Specific Notes

- Physical GPU choice is not made inside the handler.
  RunPod Serverless hardware is configured on the endpoint itself. The request-level `gpu_profile` in this repo is an optimization hint, not infrastructure provisioning.
- Large outputs should prefer bucket upload.
  `output_mode=auto` uses `bucket_url` when either RunPod bucket environment variables or top-level request `s3Config` are present.
- Models are resolved lazily.
  Missing diffusion models, support assets, and default LoRAs are downloaded on demand into `/runpod-volume` when available.
- `refresh_worker` is supported for fragmentation-sensitive or profile-switch-heavy workloads.
- Inline `base64` video output is guarded by a conservative encoded-size limit because RunPod response payload limits are much smaller than typical generated MP4 files.

## RunPod Compatibility Snapshot

Checked against RunPod Serverless docs on 2026-04-01.

- Aligned: standard handler startup, `progress_update`, `refresh_worker`, `/runpod-volume` mount assumptions.
- Aligned after local fixes: request-level `s3Config` can now be used for `bucket_url` uploads.
- Remaining operational caveat: the worker does not know whether the caller used `/run` or `/runsync`, so inline payload checks use one conservative threshold instead of endpoint-specific limits.

## Safe Change Areas

- Add new Wan 2.2 profiles in `MODEL_PROFILES`.
- Extend GPU presets in `GPU_PROFILES`.
- Adjust output handling in `build_output`.
- Tune parameter validation and defaults in `handler`.

## Caution Areas

- Do not break workflow node IDs unless the JSON workflows are updated together with the handler.
- GGUF models must keep `quantization=disabled` at request construction time because the WanVideo wrapper converts GGUF internally.
- LoRA behavior depends on ComfyUI-WanVideoWrapper semantics. GGUF and scaled FP8 should continue using unmerged LoRA behavior.

## Operational Recommendations

- Prefer mounting a RunPod network volume at `/runpod-volume`.
- Keep frequently used LoRAs under `/runpod-volume/loras`.
- If multiple hardware classes are needed in production, create multiple RunPod endpoints instead of expecting one endpoint to change hardware per job.
