import base64
import binascii
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import runpod
import websocket
from runpod.serverless.utils import rp_upload


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
CLIENT_ID = str(uuid.uuid4())
DEFAULT_OUTPUT_MODE = os.getenv("DEFAULT_OUTPUT_MODE", "auto")

COMFY_MODEL_DIR = Path("/ComfyUI/models/diffusion_models")
RUNPOD_VOLUME_MODEL_DIR = Path("/runpod-volume/models")
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", str(RUNPOD_VOLUME_MODEL_DIR if RUNPOD_VOLUME_MODEL_DIR.exists() else COMFY_MODEL_DIR)))
DEFAULT_MODEL_PROFILE = os.getenv("DEFAULT_MODEL_PROFILE", "fp8_e4m3fn")

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "fp8_e4m3fn": {
        "description": "Default scaled FP8 profile, best overall balance on 48GB+ GPUs.",
        "min_vram_gb": 48,
        "quantization": "fp8_e4m3fn_scaled",
        "files": {
            "high": {
                "filename": "Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
                "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
            },
            "low": {
                "filename": "Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors",
                "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors",
            },
        },
    },
    "fp8_e5m2": {
        "description": "Scaled FP8 e5m2 profile, useful fallback for some GPU/compiler combinations.",
        "min_vram_gb": 48,
        "quantization": "fp8_e5m2_scaled",
        "files": {
            "high": {
                "filename": "Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors",
                "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors",
            },
            "low": {
                "filename": "Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors",
                "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors",
            },
        },
    },
    "gguf_q2_k": {
        "description": "Lowest VRAM GGUF profile. Fastest to fit, lowest quality ceiling.",
        "min_vram_gb": 16,
        "quantization": "disabled",
        "files": {
            "high": {
                "filename": "wan2.2_i2v_high_noise_14B_Q2_K.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q2_K.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q2_K.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q2_K.gguf",
            },
        },
    },
    "gguf_q3_k_m": {
        "description": "Low-VRAM GGUF profile for 20-24GB class GPUs.",
        "min_vram_gb": 20,
        "quantization": "disabled",
        "files": {
            "high": {
                "filename": "wan2.2_i2v_high_noise_14B_Q3_K_M.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q3_K_M.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q3_K_M.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q3_K_M.gguf",
            },
        },
    },
    "gguf_q4_k_m": {
        "description": "Recommended GGUF profile for 24GB class GPUs.",
        "min_vram_gb": 24,
        "quantization": "disabled",
        "files": {
            "high": {
                "filename": "wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
            },
        },
    },
    "gguf_q5_k_m": {
        "description": "Higher-quality GGUF profile for 32GB+ GPUs.",
        "min_vram_gb": 32,
        "quantization": "disabled",
        "files": {
            "high": {
                "filename": "wan2.2_i2v_high_noise_14B_Q5_K_M.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q5_K_M.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q5_K_M.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q5_K_M.gguf",
            },
        },
    },
    "gguf_q6_k": {
        "description": "Near-full-quality GGUF profile for 40GB+ GPUs.",
        "min_vram_gb": 40,
        "quantization": "disabled",
        "files": {
            "high": {
                "filename": "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf",
            },
        },
    },
    "gguf_q8_0": {
        "description": "Largest GGUF profile, usually for 48GB+ GPUs.",
        "min_vram_gb": 48,
        "quantization": "disabled",
        "files": {
            "high": {
                "filename": "wan2.2_i2v_high_noise_14B_Q8_0.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q8_0.gguf",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
            },
        },
    },
}

GPU_PROFILES = {
    "L4_24GB": {"target_vram_gb": 24, "recommended_model_profile": "gguf_q4_k_m"},
    "RTX_4090_24GB": {"target_vram_gb": 24, "recommended_model_profile": "gguf_q4_k_m"},
    "A5000_24GB": {"target_vram_gb": 24, "recommended_model_profile": "gguf_q4_k_m"},
    "A40_48GB": {"target_vram_gb": 48, "recommended_model_profile": "fp8_e4m3fn"},
    "L40S_48GB": {"target_vram_gb": 48, "recommended_model_profile": "fp8_e4m3fn"},
    "RTX_PRO_6000_48GB": {"target_vram_gb": 48, "recommended_model_profile": "fp8_e4m3fn"},
    "A100_80GB": {"target_vram_gb": 80, "recommended_model_profile": "fp8_e4m3fn"},
    "H100_80GB": {"target_vram_gb": 80, "recommended_model_profile": "fp8_e4m3fn"},
}


def progress(job: Dict[str, Any], message: str) -> None:
    logger.info(message)
    try:
        runpod.serverless.progress_update(job, message)
    except Exception as exc:
        logger.warning("Failed to send progress update: %s", exc)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_nearest_multiple_of_16(value: Any) -> int:
    try:
        numeric_value = float(value)
    except Exception as exc:
        raise ValueError(f"width/height must be numeric: {value}") from exc
    adjusted = int(round(numeric_value / 16.0) * 16)
    return max(adjusted, 16)


def strip_data_uri_prefix(raw_base64: str) -> str:
    if raw_base64.startswith("data:") and "," in raw_base64:
        return raw_base64.split(",", 1)[1]
    return raw_base64


def download_to_path(url: str, output_path: Path, timeout: int = 600) -> Path:
    ensure_directory(output_path.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{output_path.name}.", dir=str(output_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        logger.info("Downloading %s -> %s", url, output_path)
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(output_path)
        return output_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def process_input(input_data: str, temp_dir: str, output_filename: str, input_type: str) -> str:
    if input_type == "path":
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Input path does not exist: {input_data}")
        logger.info("Using local path input: %s", input_data)
        return input_data
    if input_type == "url":
        output_path = ensure_directory(Path(temp_dir)) / output_filename
        return str(download_to_path(input_data, output_path))
    if input_type == "base64":
        try:
            decoded_data = base64.b64decode(strip_data_uri_prefix(input_data))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 input: {exc}") from exc
        output_path = ensure_directory(Path(temp_dir)) / output_filename
        with open(output_path, "wb") as handle:
            handle.write(decoded_data)
        return str(output_path)
    raise ValueError(f"Unsupported input type: {input_type}")


def queue_prompt(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://{SERVER_ADDRESS}:8188/prompt"
    data = json.dumps({"prompt": prompt_payload, "client_id": CLIENT_ID}).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(request).read())


def get_history(prompt_id: str) -> Dict[str, Any]:
    url = f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def wait_for_comfyui() -> None:
    http_url = f"http://{SERVER_ADDRESS}:8188/"
    for attempt in range(180):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info("ComfyUI HTTP check succeeded on attempt %s", attempt + 1)
            return
        except Exception as exc:
            if attempt == 179:
                raise RuntimeError("ComfyUI server is not reachable.") from exc
            time.sleep(1)


def collect_video_path(prompt_payload: Dict[str, Any]) -> str:
    prompt_id = queue_prompt(prompt_payload)["prompt_id"]
    ws_url = f"ws://{SERVER_ADDRESS}:8188/ws?clientId={CLIENT_ID}"
    ws = websocket.WebSocket()
    try:
        for attempt in range(36):
            try:
                ws.connect(ws_url)
                break
            except Exception as exc:
                if attempt == 35:
                    raise RuntimeError("Timed out connecting to ComfyUI websocket.") from exc
                time.sleep(5)

        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break

        history = get_history(prompt_id)[prompt_id]
        for node_output in history.get("outputs", {}).values():
            for video in node_output.get("gifs", []):
                fullpath = video.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    return fullpath
    finally:
        try:
            ws.close()
        except Exception:
            pass

    raise FileNotFoundError("Generated video file not found in ComfyUI history output.")


def load_workflow(workflow_path: str) -> Dict[str, Any]:
    with open(workflow_path, "r", encoding="utf-8") as file:
        return json.load(file)


def choose_model_profile(job_input: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    requested_profile = job_input.get("model_profile")
    if requested_profile:
        if requested_profile not in MODEL_PROFILES:
            raise ValueError(
                f"Unsupported model_profile '{requested_profile}'. "
                f"Available profiles: {', '.join(sorted(MODEL_PROFILES))}"
            )
        return requested_profile, MODEL_PROFILES[requested_profile], "explicit model_profile"

    gpu_profile = job_input.get("gpu_profile")
    if gpu_profile:
        if gpu_profile not in GPU_PROFILES:
            raise ValueError(
                f"Unsupported gpu_profile '{gpu_profile}'. "
                f"Available gpu profiles: {', '.join(sorted(GPU_PROFILES))}"
            )
        selected = GPU_PROFILES[gpu_profile]["recommended_model_profile"]
        return selected, MODEL_PROFILES[selected], f"gpu_profile={gpu_profile}"

    target_vram_gb = job_input.get("target_vram_gb")
    if target_vram_gb is not None:
        try:
            target_vram_gb = int(target_vram_gb)
        except Exception as exc:
            raise ValueError(f"target_vram_gb must be an integer, got: {target_vram_gb}") from exc

        fitting_profiles = [
            key for key, profile in MODEL_PROFILES.items()
            if profile["min_vram_gb"] <= target_vram_gb
        ]
        if not fitting_profiles:
            selected = "gguf_q2_k"
        elif target_vram_gb >= 48:
            selected = "fp8_e4m3fn"
        elif target_vram_gb >= 40:
            selected = "gguf_q6_k"
        elif target_vram_gb >= 32:
            selected = "gguf_q5_k_m"
        elif target_vram_gb >= 24:
            selected = "gguf_q4_k_m"
        elif target_vram_gb >= 20:
            selected = "gguf_q3_k_m"
        else:
            selected = "gguf_q2_k"
        return selected, MODEL_PROFILES[selected], f"target_vram_gb={target_vram_gb}"

    fallback = DEFAULT_MODEL_PROFILE if DEFAULT_MODEL_PROFILE in MODEL_PROFILES else "fp8_e4m3fn"
    return fallback, MODEL_PROFILES[fallback], f"default={fallback}"


def ensure_model_profile_available(profile_key: str, profile: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, str]:
    progress(job, f"Ensuring model profile '{profile_key}' is available")
    ensure_directory(MODEL_ROOT)
    resolved_files = {}
    for expert_name, file_info in profile["files"].items():
        target_path = MODEL_ROOT / file_info["filename"]
        if not target_path.exists():
            progress(job, f"Downloading {expert_name} expert for '{profile_key}'")
            download_to_path(file_info["url"], target_path)
        resolved_files[expert_name] = target_path.name
    return resolved_files


def apply_model_profile(prompt_payload: Dict[str, Any], profile_key: str, profile: Dict[str, Any], resolved_files: Dict[str, str]) -> None:
    for node_id, expert_name in (("122", "high"), ("549", "low")):
        prompt_payload[node_id]["inputs"]["model"] = resolved_files[expert_name]
        prompt_payload[node_id]["inputs"]["quantization"] = profile["quantization"]

    logger.info(
        "Applied model profile %s with quantization=%s, files=%s",
        profile_key,
        profile["quantization"],
        resolved_files,
    )


def resolve_output_mode(job_input: Dict[str, Any]) -> str:
    output_mode = job_input.get("output_mode", DEFAULT_OUTPUT_MODE)
    if output_mode not in {"auto", "base64", "bucket_url"}:
        raise ValueError("output_mode must be one of: auto, base64, bucket_url")

    has_bucket_config = bool(
        os.getenv("BUCKET_ENDPOINT_URL")
        and os.getenv("BUCKET_ACCESS_KEY_ID")
        and os.getenv("BUCKET_SECRET_ACCESS_KEY")
    )
    if output_mode == "auto":
        return "bucket_url" if has_bucket_config else "base64"
    if output_mode == "bucket_url" and not has_bucket_config:
        raise ValueError("output_mode='bucket_url' requires RunPod bucket environment variables.")
    return output_mode


def build_output(video_path: str, job: Dict[str, Any], job_input: Dict[str, Any], profile_key: str, selection_reason: str) -> Dict[str, Any]:
    output_mode = resolve_output_mode(job_input)
    payload: Dict[str, Any] = {
        "model_profile": profile_key,
        "model_selection_reason": selection_reason,
        "output_mode": output_mode,
    }

    if output_mode == "bucket_url":
        progress(job, "Uploading generated video to object storage")
        filename = os.path.basename(video_path)
        video_url = rp_upload.upload_file_to_bucket(
            file_name=filename,
            file_location=video_path,
            prefix=f"{job['id']}/",
        )
        payload["video_url"] = video_url
    else:
        progress(job, "Encoding generated video to base64")
        with open(video_path, "rb") as handle:
            payload["video"] = base64.b64encode(handle.read()).decode("utf-8")

    if job_input.get("include_local_path", False):
        payload["local_video_path"] = video_path

    return payload


def get_image_inputs(job_input: Dict[str, Any], task_dir: str) -> Tuple[str, Optional[str]]:
    image_path = None
    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_dir, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_dir, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_dir, "input_image.jpg", "base64")
    else:
        image_path = "/example_image.png"
        logger.info("Using bundled example image: %s", image_path)

    end_image_path = None
    if "end_image_path" in job_input:
        end_image_path = process_input(job_input["end_image_path"], task_dir, "end_image.jpg", "path")
    elif "end_image_url" in job_input:
        end_image_path = process_input(job_input["end_image_url"], task_dir, "end_image.jpg", "url")
    elif "end_image_base64" in job_input:
        end_image_path = process_input(job_input["end_image_base64"], task_dir, "end_image.jpg", "base64")

    return image_path, end_image_path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})
    logger.info("Received job input: %s", json.dumps(job_input, ensure_ascii=False))

    if job_input.get("describe_capabilities", False):
        return {
            "available_model_profiles": MODEL_PROFILES,
            "available_gpu_profiles": GPU_PROFILES,
            "default_model_profile": DEFAULT_MODEL_PROFILE,
            "default_output_mode": DEFAULT_OUTPUT_MODE,
        }

    if "prompt" not in job_input:
        return {"error": "Missing required input: prompt"}

    progress(job, "Preparing inputs")
    task_dir = f"/tmp/task_{uuid.uuid4()}"
    image_path, end_image_path = get_image_inputs(job_input, task_dir)

    profile_key, profile, selection_reason = choose_model_profile(job_input)
    resolved_files = ensure_model_profile_available(profile_key, profile, job)

    lora_pairs = job_input.get("lora_pairs", [])
    if len(lora_pairs) > 4:
        logger.warning("Received %s LoRA pairs. Only the first 4 will be used.", len(lora_pairs))
        lora_pairs = lora_pairs[:4]

    workflow_file = "/new_Wan22_flf2v_api.json" if end_image_path else "/new_Wan22_api.json"
    prompt_payload = load_workflow(workflow_file)
    apply_model_profile(prompt_payload, profile_key, profile, resolved_files)

    length = int(job_input.get("length", 81))
    steps = int(job_input.get("steps", 10))
    seed = int(job_input.get("seed", 42))
    cfg = float(job_input.get("cfg", 2.0))
    width = to_nearest_multiple_of_16(job_input.get("width", 480))
    height = to_nearest_multiple_of_16(job_input.get("height", 832))
    context_overlap = int(job_input.get("context_overlap", 48))

    prompt_payload["244"]["inputs"]["image"] = image_path
    prompt_payload["541"]["inputs"]["num_frames"] = length
    prompt_payload["135"]["inputs"]["positive_prompt"] = job_input["prompt"]
    prompt_payload["135"]["inputs"]["negative_prompt"] = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    prompt_payload["220"]["inputs"]["seed"] = seed
    prompt_payload["540"]["inputs"]["seed"] = seed
    prompt_payload["540"]["inputs"]["cfg"] = cfg
    prompt_payload["235"]["inputs"]["value"] = width
    prompt_payload["236"]["inputs"]["value"] = height
    prompt_payload["498"]["inputs"]["context_overlap"] = context_overlap
    prompt_payload["498"]["inputs"]["context_frames"] = length

    if "834" in prompt_payload:
        prompt_payload["834"]["inputs"]["steps"] = steps
        prompt_payload["829"]["inputs"]["step"] = max(1, int(steps * 0.6))

    if end_image_path:
        prompt_payload["617"]["inputs"]["image"] = end_image_path

    if lora_pairs:
        for index, lora_pair in enumerate(lora_pairs):
            high = lora_pair.get("high")
            low = lora_pair.get("low")
            if high:
                prompt_payload["279"]["inputs"][f"lora_{index + 1}"] = high
                prompt_payload["279"]["inputs"][f"strength_{index + 1}"] = lora_pair.get("high_weight", 1.0)
            if low:
                prompt_payload["553"]["inputs"][f"lora_{index + 1}"] = low
                prompt_payload["553"]["inputs"][f"strength_{index + 1}"] = lora_pair.get("low_weight", 1.0)

    progress(job, f"Running workflow with model profile '{profile_key}'")
    wait_for_comfyui()
    video_path = collect_video_path(prompt_payload)
    payload = build_output(video_path, job, job_input, profile_key, selection_reason)

    if job_input.get("refresh_worker", False):
        return {"refresh_worker": True, "job_results": payload}
    return payload


runpod.serverless.start({"handler": handler})
