import base64
import binascii
import errno
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
from typing import Any, Dict, List, Optional, Tuple

import runpod
import websocket
from runpod.serverless.utils import rp_upload


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
CLIENT_ID = str(uuid.uuid4())
DEFAULT_OUTPUT_MODE = os.getenv("DEFAULT_OUTPUT_MODE", "auto")

COMFY_MODEL_BASE = Path("/ComfyUI/models")
COMFY_LORA_DIR = COMFY_MODEL_BASE / "loras"
RUNPOD_VOLUME_BASE = Path("/runpod-volume/models")
RUNPOD_VOLUME_LORA_DIR = Path("/runpod-volume/loras")
MODEL_BASE_ROOT = Path(
    os.getenv(
        "MODEL_BASE_ROOT",
        str(RUNPOD_VOLUME_BASE if RUNPOD_VOLUME_BASE.exists() else COMFY_MODEL_BASE),
    )
)
LORA_ROOT = Path(
    os.getenv(
        "LORA_ROOT",
        str(RUNPOD_VOLUME_LORA_DIR if RUNPOD_VOLUME_LORA_DIR.exists() else COMFY_LORA_DIR),
    )
)
DEFAULT_MODEL_PROFILE = os.getenv("DEFAULT_MODEL_PROFILE", "fp8_e4m3fn")
MAX_INLINE_BASE64_BYTES = int(os.getenv("MAX_INLINE_BASE64_BYTES", "9000000"))
DEFAULT_HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
DEFAULT_CIVITAI_TOKEN = os.getenv("CIVITAI_API_TOKEN", "")
CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"
HUGGINGFACE_BASE = "https://huggingface.co"
HTTP_USER_AGENT = "generate-video-worker/1.0"
SENSITIVE_INPUT_KEYS = {"image_base64", "end_image_base64", "civitai_token", "huggingface_token"}

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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors",
            },
            "low": {
                "filename": "Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors",
            },
            "low": {
                "filename": "Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q2_K.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q2_K.gguf",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q3_K_M.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q3_K_M.gguf",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q5_K_M.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q5_K_M.gguf",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
                "subdir": "diffusion_models",
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
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
            },
            "low": {
                "filename": "wan2.2_i2v_low_noise_14B_Q8_0.gguf",
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
            },
        },
    },
}

SUPPORT_ASSETS: Dict[str, Dict[str, str]] = {
    "clip_vision": {
        "filename": "clip_vision_h.safetensors",
        "subdir": "clip_vision",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
    },
    "text_encoder": {
        "filename": "umt5-xxl-enc-bf16.safetensors",
        "subdir": "text_encoders",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
    },
    "vae": {
        "filename": "Wan2_1_VAE_bf16.safetensors",
        "subdir": "vae",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors",
    },
}

DEFAULT_LORA_ASSETS: Dict[str, Dict[str, str]] = {
    "high_noise_model.safetensors": {
        "filename": "high_noise_model.safetensors",
        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
    },
    "low_noise_model.safetensors": {
        "filename": "low_noise_model.safetensors",
        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
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


def sanitize_for_log(value: Any, key: Optional[str] = None) -> Any:
    if key in SENSITIVE_INPUT_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {k: sanitize_for_log(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, str) and len(value) > 240:
        return f"<omitted len={len(value)}>"
    return value


def make_request(url: str, headers: Optional[Dict[str, str]] = None, method: Optional[str] = None) -> urllib.request.Request:
    final_headers = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        final_headers.update(headers)
    return urllib.request.Request(url, headers=final_headers, method=method)


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
    with urllib.request.urlopen(make_request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def download_to_path(url: str, output_path: Path, timeout: int = 600, headers: Optional[Dict[str, str]] = None) -> Path:
    ensure_directory(output_path.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{output_path.name}.", dir=str(output_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        logger.info("Downloading %s -> %s", url, output_path)
        with urllib.request.urlopen(make_request(url, headers=headers), timeout=timeout) as response, open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(output_path)
        return output_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def candidate_model_targets(subdir: str, filename: str) -> Tuple[Path, ...]:
    candidates = [MODEL_BASE_ROOT / subdir / filename]
    fallback = COMFY_MODEL_BASE / subdir / filename
    if fallback not in candidates:
        candidates.append(fallback)
    return tuple(candidates)


def candidate_lora_targets(filename: str) -> Tuple[Path, ...]:
    candidates = [LORA_ROOT / filename]
    fallback = COMFY_LORA_DIR / filename
    if fallback not in candidates:
        candidates.append(fallback)
    return tuple(candidates)


def sanitize_filename(filename: str) -> str:
    cleaned = os.path.basename((filename or "").strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid filename: {filename!r}")
    return cleaned


def infer_filename_from_url(url: str, fallback_name: str) -> str:
    path = urllib.parse.urlparse(url).path
    candidate = os.path.basename(path)
    if candidate and "." in candidate:
        return sanitize_filename(candidate)
    return sanitize_filename(fallback_name)


def get_request_token(job_input: Dict[str, Any], field_name: str, env_value: str) -> Optional[str]:
    token = str(job_input.get(field_name) or env_value or "").strip()
    return token or None


def make_bearer_headers(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def append_query_param(url: str, key: str, value: Optional[str]) -> str:
    if not value:
        return url
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query[key] = [value]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def extract_filename_from_content_disposition(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    parts = [part.strip() for part in header_value.split(";")]
    for part in parts:
        if part.lower().startswith("filename="):
            value = part.split("=", 1)[1].strip().strip('"')
            if value:
                return sanitize_filename(value)
    return None


def head_filename(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
    try:
        with urllib.request.urlopen(make_request(url, headers=headers, method="HEAD"), timeout=30) as response:
            return extract_filename_from_content_disposition(response.headers.get("content-disposition"))
    except Exception:
        return None


def ensure_asset_available(asset: Dict[str, Any], target_paths: Tuple[Path, ...], progress_message: Optional[str] = None) -> str:
    for target_path in target_paths:
        if target_path.exists():
            return target_path.name

    if progress_message:
        logger.info(progress_message)

    last_error: Optional[Exception] = None
    for index, target_path in enumerate(target_paths):
        try:
            ensure_directory(target_path.parent)
            download_to_path(asset["url"], target_path, headers=asset.get("headers"))
            return target_path.name
        except OSError as exc:
            last_error = exc
            is_last_target = index == len(target_paths) - 1
            if exc.errno == errno.ENOSPC and not is_last_target:
                logger.warning(
                    "No space left while downloading %s to %s. Retrying with fallback path %s",
                    asset["filename"],
                    target_path,
                    target_paths[index + 1],
                )
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to ensure asset is available: {asset['filename']}")


def ensure_support_assets(job: Dict[str, Any]) -> None:
    progress(job, "Ensuring shared Wan support assets are available")
    for asset_name, asset in SUPPORT_ASSETS.items():
        target_paths = candidate_model_targets(asset["subdir"], asset["filename"])
        ensure_asset_available(asset, target_paths, f"Downloading support asset '{asset_name}'")


def ensure_default_loras(job: Dict[str, Any]) -> None:
    progress(job, "Ensuring default lightning LoRAs are available")
    for lora_name, asset in DEFAULT_LORA_ASSETS.items():
        target_paths = candidate_lora_targets(asset["filename"])
        ensure_asset_available(asset, target_paths, f"Downloading default LoRA '{lora_name}'")


def resolve_existing_lora(filename: str) -> str:
    resolved_name = sanitize_filename(filename)
    for target_path in candidate_lora_targets(resolved_name):
        if target_path.exists():
            return resolved_name
    raise FileNotFoundError(
        f"LoRA file '{resolved_name}' was not found. Place it under '{LORA_ROOT}' "
        "or provide a Civitai/Hugging Face/direct download reference in `loras`."
    )


def ensure_custom_lora(url: str, filename_hint: Optional[str], index: int, headers: Optional[Dict[str, str]] = None) -> str:
    filename = sanitize_filename(filename_hint) if filename_hint else infer_filename_from_url(url, f"custom_lora_{index + 1}.safetensors")
    target_paths = candidate_lora_targets(filename)
    asset = {"filename": filename, "url": url, "headers": headers or {}}
    ensure_asset_available(asset, target_paths, f"Downloading custom LoRA '{filename}'")
    return filename


def build_huggingface_resolve_url(repo: str, path: str, revision: str = "main") -> str:
    repo = repo.strip("/")
    path = path.strip("/")
    revision = revision.strip("/") or "main"
    if not repo or not path:
        raise ValueError("Hugging Face LoRA references require both `repo` and `path`.")
    return f"{HUGGINGFACE_BASE}/{repo}/resolve/{revision}/{path}"


def parse_huggingface_url(url: str) -> Tuple[str, str, str]:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
        raise ValueError(
            "Unsupported Hugging Face URL. Use a file URL like "
            "`https://huggingface.co/<repo>/resolve/<revision>/<path>` or `/blob/...`."
        )
    repo = "/".join(parts[:2])
    revision = parts[3]
    path = "/".join(parts[4:])
    return repo, revision, path


def resolve_huggingface_reference(entry: Dict[str, Any], source: str, index: int, token: Optional[str]) -> Dict[str, Any]:
    if source:
        repo, revision, path = parse_huggingface_url(source)
    else:
        repo = str(entry.get("repo", "")).strip()
        revision = str(entry.get("revision", "main")).strip() or "main"
        path = str(entry.get("path", "")).strip()
    url = build_huggingface_resolve_url(repo, path, revision)
    filename_hint = str(entry.get("filename") or entry.get("name") or "").strip() or os.path.basename(path)
    return {
        "provider": "huggingface",
        "download_url": url,
        "filename": sanitize_filename(filename_hint),
        "headers": make_bearer_headers(token),
        "details": {"repo": repo, "revision": revision, "path": path},
    }


def choose_civitai_file(model_data: Dict[str, Any], version_id: Optional[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    versions = model_data.get("modelVersions") or []
    if not versions:
        raise ValueError(f"Civitai model {model_data.get('id')} has no downloadable versions.")
    chosen_version = None
    if version_id is not None:
        for version in versions:
            if int(version.get("id")) == int(version_id):
                chosen_version = version
                break
        if chosen_version is None:
            raise ValueError(f"Civitai model {model_data.get('id')} does not contain version {version_id}.")
    else:
        chosen_version = versions[0]
    files = chosen_version.get("files") or []
    if not files:
        raise ValueError(f"Civitai model version {chosen_version.get('id')} has no downloadable files.")
    safetensor_files = [file_info for file_info in files if str(file_info.get("name", "")).lower().endswith(".safetensors")]
    return chosen_version, safetensor_files[0] if safetensor_files else files[0]


def resolve_civitai_reference(entry: Dict[str, Any], source: str, index: int, token: Optional[str]) -> Dict[str, Any]:
    parsed = urllib.parse.urlparse(source) if source else None
    model_id = entry.get("model_id") or entry.get("modelId")
    version_id = entry.get("model_version_id") or entry.get("modelVersionId")

    if parsed and parsed.netloc.endswith("civitai.com"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "download" and parts[2] == "models":
            version_id = version_id or parts[3]
        elif len(parts) >= 2 and parts[0] == "models":
            model_id = model_id or parts[1]
            query = urllib.parse.parse_qs(parsed.query)
            version_id = version_id or (query.get("modelVersionId", [None])[0])

    if model_id:
        model_url = f"{CIVITAI_API_BASE}/models/{int(model_id)}"
        model_data = fetch_json(model_url, headers=make_bearer_headers(token))
        chosen_version, file_info = choose_civitai_file(model_data, int(version_id) if version_id else None)
        download_url = file_info["downloadUrl"]
        filename = sanitize_filename(file_info["name"])
        resolved_version_id = int(chosen_version["id"])
        resolved_model_id = int(model_data["id"])
    elif version_id:
        resolved_version_id = int(version_id)
        download_url = f"{CIVITAI_DOWNLOAD_BASE}/{resolved_version_id}"
        filename = (
            str(entry.get("filename") or entry.get("name") or "").strip()
            or head_filename(append_query_param(download_url, "token", token), headers=make_bearer_headers(token))
            or f"civitai_lora_{resolved_version_id}.safetensors"
        )
        filename = sanitize_filename(filename)
        resolved_model_id = None
    else:
        raise ValueError(
            "Unsupported Civitai reference. Use a model page URL with `modelVersionId`, "
            "a download URL, or structured `modelId`/`modelVersionId`."
        )

    download_url = append_query_param(download_url, "token", token)
    return {
        "provider": "civitai",
        "download_url": download_url,
        "filename": filename,
        "headers": make_bearer_headers(token),
        "details": {"model_id": resolved_model_id, "model_version_id": resolved_version_id},
    }


def resolve_lora_source(entry: Dict[str, Any], source: str, index: int, job_input: Dict[str, Any]) -> Dict[str, Any]:
    source = source.strip()
    if not source:
        raise ValueError("LoRA source cannot be empty.")

    if source.startswith("http://") or source.startswith("https://"):
        parsed = urllib.parse.urlparse(source)
        host = parsed.netloc.lower()
        if host.endswith("huggingface.co"):
            token = str(entry.get("huggingface_token") or entry.get("token") or get_request_token(job_input, "huggingface_token", DEFAULT_HUGGINGFACE_TOKEN) or "").strip() or None
            return resolve_huggingface_reference(entry, source, index, token)
        if host.endswith("civitai.com"):
            token = str(entry.get("civitai_token") or entry.get("token") or get_request_token(job_input, "civitai_token", DEFAULT_CIVITAI_TOKEN) or "").strip() or None
            return resolve_civitai_reference(entry, source, index, token)
        filename = str(entry.get("filename") or entry.get("name") or "").strip()
        if not filename:
            filename = head_filename(source) or infer_filename_from_url(source, f"custom_lora_{index + 1}.safetensors")
        return {
            "provider": "direct_url",
            "download_url": source,
            "filename": sanitize_filename(filename),
            "headers": entry.get("headers") or {},
            "details": {"url": source},
        }

    return {
        "provider": "existing_file",
        "download_url": "",
        "filename": resolve_existing_lora(source),
        "headers": {},
        "details": {"filename": sanitize_filename(source)},
    }


def normalize_single_lora(entry: Any, index: int, job_input: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(entry, str):
        entry_dict: Dict[str, Any] = {"source": entry}
    elif isinstance(entry, dict):
        entry_dict = dict(entry)
    else:
        raise ValueError("Each `loras` entry must be a string or object.")

    weight = float(entry_dict.get("weight", 1.0))
    source = str(entry_dict.get("source") or "").strip()
    filename_only = str(entry_dict.get("filename") or entry_dict.get("name") or "").strip()

    if source:
        resolved = resolve_lora_source(entry_dict, source, index, job_input)
    elif entry_dict.get("url"):
        entry_dict["source"] = str(entry_dict["url"]).strip()
        resolved = resolve_lora_source(entry_dict, entry_dict["source"], index, job_input)
    elif entry_dict.get("repo") and entry_dict.get("path"):
        token = str(entry_dict.get("huggingface_token") or entry_dict.get("token") or get_request_token(job_input, "huggingface_token", DEFAULT_HUGGINGFACE_TOKEN) or "").strip() or None
        resolved = resolve_huggingface_reference(entry_dict, "", index, token)
    elif any(entry_dict.get(key) for key in ("modelId", "model_id", "modelVersionId", "model_version_id")):
        token = str(entry_dict.get("civitai_token") or entry_dict.get("token") or get_request_token(job_input, "civitai_token", DEFAULT_CIVITAI_TOKEN) or "").strip() or None
        resolved = resolve_civitai_reference(entry_dict, "", index, token)
    elif filename_only:
        resolved = resolve_lora_source(entry_dict, filename_only, index, job_input)
    else:
        raise ValueError("Each `loras` entry must include `source`, `url`, `filename`, or provider-specific fields.")

    resolved_filename = (
        resolved["filename"]
        if not resolved["download_url"]
        else ensure_custom_lora(resolved["download_url"], resolved["filename"], index, headers=resolved["headers"])
    )
    return (
        {
            "high": resolved_filename,
            "low": resolved_filename,
            "high_weight": weight,
            "low_weight": weight,
        },
        {
            "provider": resolved["provider"],
            "input": source or filename_only or resolved["download_url"],
            "resolved_filename": resolved_filename,
            "weight": weight,
            **resolved["details"],
        },
    )


def normalize_legacy_branch(source: str, weight: float, index: int, job_input: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
    source = str(source or "").strip()
    if not source:
        return None, None, weight
    resolved = resolve_lora_source({}, source, index, job_input)
    filename = resolved["filename"] if not resolved["download_url"] else ensure_custom_lora(
        resolved["download_url"], resolved["filename"], index, headers=resolved["headers"]
    )
    return filename, {
        "provider": resolved["provider"],
        "input": source,
        "resolved_filename": filename,
        "weight": weight,
        **resolved["details"],
    }, weight


def normalize_lora_pairs(job_input: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized: List[Dict[str, Any]] = []
    resolved_metadata: List[Dict[str, Any]] = []

    generic_loras = job_input.get("loras", [])
    if generic_loras:
        if not isinstance(generic_loras, list):
            raise ValueError("`loras` must be a list.")
        for index, lora in enumerate(generic_loras):
            pair, metadata = normalize_single_lora(lora, index, job_input)
            normalized.append(pair)
            resolved_metadata.append(metadata)

    legacy_lora_pairs = job_input.get("lora_pairs", [])
    if legacy_lora_pairs:
        if not isinstance(legacy_lora_pairs, list):
            raise ValueError("`lora_pairs` must be a list of objects.")
        for index, lora_pair in enumerate(legacy_lora_pairs, start=len(normalized)):
            if not isinstance(lora_pair, dict):
                raise ValueError("Each `lora_pairs` entry must be an object.")

            high_source = str(lora_pair.get("high_url") or lora_pair.get("high_source") or lora_pair.get("high") or "").strip()
            low_source = str(lora_pair.get("low_url") or lora_pair.get("low_source") or lora_pair.get("low") or "").strip()
            high_weight = float(lora_pair.get("high_weight", 1.0))
            low_weight = float(lora_pair.get("low_weight", 1.0))

            high, high_meta, _ = normalize_legacy_branch(high_source, high_weight, index, job_input)
            low, low_meta, _ = normalize_legacy_branch(low_source, low_weight, index, job_input)

            if not high and not low:
                raise ValueError("Each `lora_pairs` entry must include at least one LoRA source.")

            normalized.append({
                "high": high,
                "low": low,
                "high_weight": high_weight,
                "low_weight": low_weight,
            })
            resolved_metadata.append({
                "provider": "legacy_pair",
                "high": high_meta,
                "low": low_meta,
            })

    return normalized, resolved_metadata


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
        logger.warning("`gpu_profile` is deprecated as a request field. Prefer `model_profile` or endpoint defaults.")
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
    resolved_files = {}
    for expert_name, file_info in profile["files"].items():
        target_paths = candidate_model_targets(file_info["subdir"], file_info["filename"])
        ensure_asset_available(
            file_info,
            target_paths,
            f"Downloading {expert_name} expert for '{profile_key}'",
        )
        resolved_files[expert_name] = file_info["filename"]
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


def get_bucket_config(job: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    request_bucket = job.get("s3Config") or {}
    if request_bucket:
        required = {"accessId", "accessSecret", "bucketName", "endpointUrl"}
        missing = sorted(required - set(request_bucket))
        if missing:
            raise ValueError(
                "s3Config is missing required fields: "
                + ", ".join(missing)
            )
        return (
            {
                "accessId": request_bucket["accessId"],
                "accessSecret": request_bucket["accessSecret"],
                "endpointUrl": request_bucket["endpointUrl"],
            },
            request_bucket["bucketName"],
        )

    has_bucket_env = bool(
        os.getenv("BUCKET_ENDPOINT_URL")
        and os.getenv("BUCKET_ACCESS_KEY_ID")
        and os.getenv("BUCKET_SECRET_ACCESS_KEY")
    )
    if has_bucket_env:
        return None, None
    return None, None


def resolve_output_mode(job: Dict[str, Any], job_input: Dict[str, Any]) -> str:
    output_mode = job_input.get("output_mode", DEFAULT_OUTPUT_MODE)
    if output_mode not in {"auto", "base64", "bucket_url"}:
        raise ValueError("output_mode must be one of: auto, base64, bucket_url")

    bucket_creds, bucket_name = get_bucket_config(job)
    has_bucket_config = bool(bucket_creds or bucket_name or (
        os.getenv("BUCKET_ENDPOINT_URL")
        and os.getenv("BUCKET_ACCESS_KEY_ID")
        and os.getenv("BUCKET_SECRET_ACCESS_KEY")
    ))
    if output_mode == "auto":
        return "bucket_url" if has_bucket_config else "base64"
    if output_mode == "bucket_url" and not has_bucket_config:
        raise ValueError(
            "output_mode='bucket_url' requires either top-level s3Config or "
            "RunPod bucket environment variables."
        )
    return output_mode


def ensure_inline_output_size(video_path: str) -> None:
    video_size = os.path.getsize(video_path)
    encoded_size = ((video_size + 2) // 3) * 4
    if encoded_size > MAX_INLINE_BASE64_BYTES:
        raise ValueError(
            "Generated video is too large for inline base64 output under conservative "
            f"RunPod payload limits (encoded_size={encoded_size} bytes, "
            f"limit={MAX_INLINE_BASE64_BYTES} bytes). Use output_mode='bucket_url' "
            "with endpoint bucket environment variables or top-level s3Config."
        )


def build_output(
    video_path: str,
    job: Dict[str, Any],
    job_input: Dict[str, Any],
    profile_key: str,
    selection_reason: str,
    resolved_loras: List[Dict[str, Any]],
) -> Dict[str, Any]:
    output_mode = resolve_output_mode(job, job_input)
    payload: Dict[str, Any] = {
        "model_profile": profile_key,
        "model_selection_reason": selection_reason,
        "output_mode": output_mode,
    }
    if resolved_loras:
        payload["resolved_loras"] = resolved_loras

    if output_mode == "bucket_url":
        progress(job, "Uploading generated video to object storage")
        filename = os.path.basename(video_path)
        bucket_creds, bucket_name = get_bucket_config(job)
        video_url = rp_upload.upload_file_to_bucket(
            file_name=filename,
            file_location=video_path,
            bucket_creds=bucket_creds,
            bucket_name=bucket_name,
            prefix=f"{job['id']}/",
        )
        payload["video_url"] = video_url
    else:
        ensure_inline_output_size(video_path)
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
    logger.info("Received job input: %s", json.dumps(sanitize_for_log(job_input), ensure_ascii=False))

    if job_input.get("describe_capabilities", False):
        return {
            "available_model_profiles": MODEL_PROFILES,
            "available_gpu_profiles": GPU_PROFILES,
            "default_model_profile": DEFAULT_MODEL_PROFILE,
            "default_output_mode": DEFAULT_OUTPUT_MODE,
            "deprecated_request_fields": ["gpu_profile"],
            "supported_lora_fields": {
                "loras": {
                    "description": "preferred; each entry may be a string or object",
                    "accepted_forms": [
                        "existing filename under /runpod-volume/loras",
                        "direct URL to a .safetensors file",
                        "Civitai model page URL with modelVersionId",
                        "Civitai download URL",
                        "Hugging Face /resolve/ or /blob/ file URL",
                        "structured Hugging Face repo/path/revision object",
                        "structured Civitai modelId/modelVersionId object",
                    ],
                    "examples": [
                        {"source": "my_style_lora.safetensors", "weight": 0.8},
                        {"source": "https://civitai.com/models/122359/detail-tweaker-xl?modelVersionId=135867", "weight": 0.8},
                        {"source": "https://huggingface.co/owner/repo/resolve/main/loras/my_style.safetensors", "weight": 0.8},
                        {"provider": "huggingface", "repo": "owner/repo", "path": "loras/my_style.safetensors", "revision": "main", "weight": 0.8},
                        {"provider": "civitai", "modelId": 122359, "modelVersionId": 135867, "weight": 0.8},
                    ],
                },
                "lora_pairs": "advanced legacy form; accepts high/low filenames or URLs and optional branch-specific weights",
            },
            "supported_secret_fields": ["civitai_token", "huggingface_token"],
        }

    if "prompt" not in job_input:
        return {"error": "Missing required input: prompt"}

    progress(job, "Preparing inputs")
    task_dir = f"/tmp/task_{uuid.uuid4()}"
    image_path, end_image_path = get_image_inputs(job_input, task_dir)

    ensure_support_assets(job)
    ensure_default_loras(job)
    profile_key, profile, selection_reason = choose_model_profile(job_input)
    resolved_files = ensure_model_profile_available(profile_key, profile, job)

    lora_pairs, resolved_loras = normalize_lora_pairs(job_input)
    if len(lora_pairs) > 4:
        logger.warning("Received %s LoRA pairs. Only the first 4 will be used.", len(lora_pairs))
        lora_pairs = lora_pairs[:4]
        resolved_loras = resolved_loras[:4]

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
    payload = build_output(video_path, job, job_input, profile_key, selection_reason, resolved_loras)

    if job_input.get("refresh_worker", False):
        return {"refresh_worker": True, "job_results": payload}
    return payload


runpod.serverless.start({"handler": handler})
