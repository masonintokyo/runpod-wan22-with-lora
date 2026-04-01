#!/usr/bin/env python3
"""
Local UI for submitting image-to-video jobs to a RunPod endpoint.

Setup:
  uv sync
  export RUNPOD_API_KEY=...
  export RUNPOD_ENDPOINT_ID=...
  uv run python local_ui_server.py
"""

import html
import json
import os
import tempfile

from flask import Flask, request, Response

from generate_video_client import GenerateVideoClient


app = Flask(__name__)

HOST = os.getenv("LOCAL_UI_HOST", "127.0.0.1")
PORT = int(os.getenv("LOCAL_UI_PORT", "8787"))
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

MODEL_OPTIONS = [
    "fp8_e4m3fn",
    "fp8_e5m2",
    "gguf_q2_k",
    "gguf_q3_k_m",
    "gguf_q4_k_m",
    "gguf_q5_k_m",
    "gguf_q6_k",
    "gguf_q8_0",
]

GPU_OPTIONS = [
    "L4_24GB",
    "RTX_4090_24GB",
    "A5000_24GB",
    "A40_48GB",
    "L40S_48GB",
    "RTX_PRO_6000_48GB",
    "A100_80GB",
    "H100_80GB",
]

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def render_options(options, selected):
    parts = ['<option value=""></option>']
    for option in options:
        attr = " selected" if option == selected else ""
        parts.append(f'<option value="{html.escape(option)}"{attr}>{html.escape(option)}</option>')
    return "".join(parts)


def render_page(values=None, result=None, error=None):
    values = values or {}
    model_options = render_options(MODEL_OPTIONS, values.get("model_profile", "gguf_q4_k_m"))
    gpu_options = render_options(GPU_OPTIONS, values.get("gpu_profile", ""))
    output_mode = values.get("output_mode", "auto")
    result_block = ""
    if error:
        result_block = f'<div class="error"><strong>Error:</strong> {html.escape(error)}</div>'
    elif result:
        result_block = "<div class='result'><h2>Result</h2>"
        result_block += f"<pre>{html.escape(json.dumps(result, indent=2, ensure_ascii=False))}</pre>"
        payload = result.get("output") or {}
        if "job_results" in payload:
            payload = payload["job_results"]
        video_url = payload.get("video_url")
        if video_url:
            result_block += f'<p><a href="{html.escape(video_url)}" target="_blank" rel="noreferrer">Open video URL</a></p>'
        result_block += "</div>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wan 2.2 RunPod UI</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: rgba(255,255,255,0.82);
      --ink: #1e1c19;
      --muted: #645d53;
      --line: #d6cfc2;
      --accent: #b24a2d;
      --accent-2: #3c6e71;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(178,74,45,0.18), transparent 30%),
        radial-gradient(circle at bottom right, rgba(60,110,113,0.18), transparent 28%),
        linear-gradient(135deg, #efe9dc, #f7f4ed 40%, #ece4d7);
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 42px; line-height: 1.05; }}
    h2 {{ margin-top: 0; }}
    p.lead {{ margin: 0 0 24px; color: var(--muted); max-width: 800px; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.45);
      backdrop-filter: blur(10px);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 16px 40px rgba(65,50,30,0.09);
    }}
    label {{ display: block; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 6px; }}
    input, textarea, select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.9);
      font: inherit;
      color: var(--ink);
    }}
    textarea {{ min-height: 100px; resize: vertical; }}
    .row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }}
    .row3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }}
    .field {{ margin-bottom: 14px; }}
    .check {{ display: flex; align-items: center; gap: 8px; margin: 10px 0; }}
    .check input {{ width: auto; }}
    button {{
      width: 100%;
      padding: 14px 18px;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), #c86a36);
      color: white;
      font-size: 16px;
      cursor: pointer;
    }}
    .hint {{ font-size: 13px; color: var(--muted); }}
    .persist-note {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
      font-size: 13px;
      color: var(--muted);
    }}
    .result, .error {{
      margin-top: 20px;
      padding: 16px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
    }}
    .error {{ border-color: #c77b6b; color: #7f2312; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    ul {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 900px) {{
      .grid, .row, .row3 {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 32px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Wan 2.2 RunPod Local UI</h1>
    <p class="lead">Upload an image, choose a model profile, and submit a job to your RunPod Serverless endpoint. Use either <code>model_profile</code> or <code>gpu_profile</code>/<code>target_vram_gb</code>.</p>
    <div class="grid">
      <div class="card">
        <form id="job-form" action="/submit" method="post" enctype="multipart/form-data">
          <div class="field">
            <label>Endpoint ID</label>
            <input name="runpod_endpoint_id" value="{html.escape(values.get("runpod_endpoint_id", RUNPOD_ENDPOINT_ID))}" required>
          </div>
          <div class="field">
            <label>API Key</label>
            <input type="password" name="runpod_api_key" value="{html.escape(values.get("runpod_api_key", RUNPOD_API_KEY))}" required>
          </div>
          <label class="check"><input type="checkbox" name="remember_api_key"> Remember API key in this browser</label>
          <div class="field">
            <label>Prompt</label>
            <textarea name="prompt" required>{html.escape(values.get("prompt", "cinematic portrait, subtle head turn, natural motion, realistic lighting"))}</textarea>
          </div>
          <div class="field">
            <label>Negative Prompt</label>
            <textarea name="negative_prompt">{html.escape(values.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))}</textarea>
          </div>
          <div class="field">
            <label>Image</label>
            <input type="file" name="image_file" accept="image/*" required>
          </div>
          <div class="row3">
            <div class="field">
              <label>Width</label>
              <input name="width" value="{html.escape(values.get("width", "480"))}">
            </div>
            <div class="field">
              <label>Height</label>
              <input name="height" value="{html.escape(values.get("height", "832"))}">
            </div>
            <div class="field">
              <label>Frames</label>
              <input name="length" value="{html.escape(values.get("length", "81"))}">
            </div>
          </div>
          <div class="row3">
            <div class="field">
              <label>Steps</label>
              <input name="steps" value="{html.escape(values.get("steps", "10"))}">
            </div>
            <div class="field">
              <label>CFG</label>
              <input name="cfg" value="{html.escape(values.get("cfg", "2.0"))}">
            </div>
            <div class="field">
              <label>Seed</label>
              <input name="seed" value="{html.escape(values.get("seed", "42"))}">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label>Model Profile</label>
              <select name="model_profile">{model_options}</select>
            </div>
            <div class="field">
              <label>GPU Profile</label>
              <select name="gpu_profile">{gpu_options}</select>
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label>Target VRAM GB</label>
              <input name="target_vram_gb" value="{html.escape(values.get("target_vram_gb", ""))}">
            </div>
            <div class="field">
              <label>Output Mode</label>
              <select name="output_mode">
                <option value="auto" {"selected" if output_mode == "auto" else ""}>auto</option>
                <option value="base64" {"selected" if output_mode == "base64" else ""}>base64</option>
                <option value="bucket_url" {"selected" if output_mode == "bucket_url" else ""}>bucket_url</option>
              </select>
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label>High LoRA File Name</label>
              <input name="lora_high" value="{html.escape(values.get("lora_high", ""))}" placeholder="example_high.safetensors">
            </div>
            <div class="field">
              <label>Low LoRA File Name</label>
              <input name="lora_low" value="{html.escape(values.get("lora_low", ""))}" placeholder="example_low.safetensors">
            </div>
          </div>
          <div class="row">
            <div class="field">
              <label>High LoRA Weight</label>
              <input name="lora_high_weight" value="{html.escape(values.get("lora_high_weight", "1.0"))}">
            </div>
            <div class="field">
              <label>Low LoRA Weight</label>
              <input name="lora_low_weight" value="{html.escape(values.get("lora_low_weight", "1.0"))}">
            </div>
          </div>
          <label class="check"><input type="checkbox" name="refresh_worker" {"checked" if values.get("refresh_worker") else ""}> Refresh worker after completion</label>
          <div class="persist-note">
            Endpoint ID and generation settings are saved in this browser automatically. API key is only saved if you enable "Remember API key in this browser".
          </div>
          <button type="submit">Submit Job</button>
        </form>
        {result_block}
      </div>
      <div class="card">
        <h2>Practical Defaults</h2>
        <ul>
          <li>24GB GPU class: use <code>gguf_q4_k_m</code>.</li>
          <li>48GB+ GPU class: use <code>fp8_e4m3fn</code>.</li>
          <li>If you do not know the hardware, leave <code>model_profile</code> empty and set <code>target_vram_gb</code>.</li>
          <li>Image-to-video requires an input image.</li>
          <li>Default lightning LoRAs are fetched lazily as <code>high_noise_model.safetensors</code> and <code>low_noise_model.safetensors</code>.</li>
          <li>Custom LoRAs should be placed in <code>/runpod-volume/loras</code>.</li>
        </ul>
        <h2>LoRA Notes</h2>
        <p class="hint">Start with <code>1.0</code> for both weights. If motion becomes unstable, reduce toward <code>0.7</code> to <code>0.9</code>. If the LoRA is very stylized, keep it below <code>0.8</code> first.</p>
        <h2>Prompt Notes</h2>
        <p class="hint">Prompts work best when they describe motion and camera behavior. Example: <code>subtle head turn, gentle blinking, cinematic lighting, natural motion</code>.</p>
      </div>
    </div>
  </div>
  <script>
    (() => {{
      const storageKey = "wan22-runpod-ui-settings-v1";
      const form = document.getElementById("job-form");
      if (!form) return;

      const persistentFields = [
        "runpod_endpoint_id",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "length",
        "steps",
        "cfg",
        "seed",
        "model_profile",
        "gpu_profile",
        "target_vram_gb",
        "output_mode",
        "lora_high",
        "lora_low",
        "lora_high_weight",
        "lora_low_weight",
        "refresh_worker",
      ];
      const sensitiveField = "runpod_api_key";
      const rememberField = "remember_api_key";

      function loadSettings() {{
        try {{
          return JSON.parse(localStorage.getItem(storageKey) || "{{}}");
        }} catch (_error) {{
          return {{}};
        }}
      }}

      function saveSettings() {{
        const saved = {{}};
        for (const name of persistentFields) {{
          const field = form.elements.namedItem(name);
          if (!field) continue;
          saved[name] = field.type === "checkbox" ? field.checked : field.value;
        }}

        const rememberApiKey = form.elements.namedItem(rememberField)?.checked;
        saved[rememberField] = Boolean(rememberApiKey);
        if (rememberApiKey) {{
          saved[sensitiveField] = form.elements.namedItem(sensitiveField)?.value || "";
        }}

        localStorage.setItem(storageKey, JSON.stringify(saved));
      }}

      function restoreSettings() {{
        const saved = loadSettings();
        for (const name of persistentFields) {{
          if (!(name in saved)) continue;
          const field = form.elements.namedItem(name);
          if (!field) continue;
          if (field.type === "checkbox") {{
            field.checked = Boolean(saved[name]);
          }} else if (!field.value) {{
            field.value = saved[name];
          }}
        }}

        const rememberApiKey = Boolean(saved[rememberField]);
        const rememberCheckbox = form.elements.namedItem(rememberField);
        if (rememberCheckbox) {{
          rememberCheckbox.checked = rememberApiKey;
        }}

        const apiKeyField = form.elements.namedItem(sensitiveField);
        if (rememberApiKey && apiKeyField && saved[sensitiveField] && !apiKeyField.value) {{
          apiKeyField.value = saved[sensitiveField];
        }}
      }}

      restoreSettings();

      form.addEventListener("input", saveSettings);
      form.addEventListener("change", saveSettings);
      form.addEventListener("submit", saveSettings);

      const rememberCheckbox = form.elements.namedItem(rememberField);
      if (rememberCheckbox) {{
        rememberCheckbox.addEventListener("change", () => {{
          if (!rememberCheckbox.checked) {{
            const saved = loadSettings();
            delete saved[sensitiveField];
            saved[rememberField] = false;
            localStorage.setItem(storageKey, JSON.stringify(saved));
          }}
        }});
      }}
    }})();
  </script>
</body>
</html>"""


@app.get("/")
def index():
    return Response(render_page(), mimetype="text/html")


@app.post("/submit")
def submit():
    values = {key: value for key, value in request.form.items()}
    image_file = request.files.get("image_file")
    if image_file is None or not image_file.filename:
        return Response(render_page(values=values, error="Image file is required."), mimetype="text/html")

    endpoint_id = values.get("runpod_endpoint_id", "").strip()
    api_key = values.get("runpod_api_key", "").strip()
    if not endpoint_id or not api_key:
        return Response(render_page(values=values, error="Endpoint ID and API key are required."), mimetype="text/html")

    suffix = os.path.splitext(image_file.filename)[1] or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            image_file.save(temp)
            temp_path = temp.name

        client = GenerateVideoClient(runpod_endpoint_id=endpoint_id, runpod_api_key=api_key)
        kwargs = {
            "image_path": temp_path,
            "prompt": values.get("prompt", "").strip(),
            "negative_prompt": values.get("negative_prompt", "").strip() or None,
            "width": int(values.get("width") or 480),
            "height": int(values.get("height") or 832),
            "length": int(values.get("length") or 81),
            "steps": int(values.get("steps") or 10),
            "seed": int(values.get("seed") or 42),
            "cfg": float(values.get("cfg") or 2.0),
            "output_mode": values.get("output_mode") or "auto",
            "refresh_worker": bool(values.get("refresh_worker")),
        }

        if values.get("model_profile"):
            kwargs["model_profile"] = values["model_profile"]
        if values.get("gpu_profile"):
            kwargs["gpu_profile"] = values["gpu_profile"]
        if values.get("target_vram_gb"):
            kwargs["target_vram_gb"] = int(values["target_vram_gb"])

        lora_high = values.get("lora_high", "").strip()
        lora_low = values.get("lora_low", "").strip()
        if lora_high or lora_low:
            kwargs["lora_pairs"] = [{
                "high": lora_high,
                "low": lora_low,
                "high_weight": float(values.get("lora_high_weight") or 1.0),
                "low_weight": float(values.get("lora_low_weight") or 1.0),
            }]

        result = client.create_video_from_image(**kwargs)
        return Response(render_page(values=values, result=result), mimetype="text/html")
    except Exception as exc:
        return Response(render_page(values=values, error=str(exc)), mimetype="text/html")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
