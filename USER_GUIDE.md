# USER GUIDE

## これは何か

このリポジトリは、Wan 2.2 を使って画像から動画を生成する `runpod.io` Serverless Worker です。内部では ComfyUI を起動し、RunPod の `/run` API からジョブを受け取ります。

## できること

- 通常の Wan 2.2 FP8 モデルを使う
- Wan 2.2 の GGUF 量子化モデルを使う
- `model_profile` を明示してモデルを切り替える
- `gpu_profile` または `target_vram_gb` から自動で適切なモデルを選ばせる
- 出力を `base64` か `bucket_url` で返す

## 重要な前提

`gpu_profile` は RunPod の実機 GPU を API リクエストだけで切り替えるものではありません。

RunPod Serverless では、実際の GPU 種別は Endpoint 設定で決まります。このリポジトリの `gpu_profile` は「このクラスの GPU / VRAM を想定して、どの Wan 2.2 モデルを使うか」を選ぶための入力です。

## 主な入力パラメータ

- `prompt`
- `image_path` / `image_url` / `image_base64`
- `end_image_path` / `end_image_url` / `end_image_base64`
- `model_profile`
- `gpu_profile`
- `target_vram_gb`
- `output_mode`
- `refresh_worker`

## 推奨モデルの目安

- 24GB クラス: `gguf_q4_k_m`
- 32GB クラス: `gguf_q5_k_m`
- 40GB クラス: `gguf_q6_k`
- 48GB 以上: `fp8_e4m3fn`

これは目安です。解像度、フレーム数、LoRA 使用有無で必要 VRAM は変わります。

## リクエスト例

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

## 実際の使い方

1. このリポジトリを GitHub に push する
2. RunPod Serverless で endpoint を作る
3. 可能なら `/runpod-volume` に network volume を付ける
4. deploy する
5. 最初のリクエストを送る

最初のリクエストは cold start になりやすいです。

## 最初の1本目の推奨

- 24GB クラスの endpoint: `model_profile = "gguf_q4_k_m"`
- 48GB 以上の endpoint: `model_profile = "fp8_e4m3fn"`
- まずは custom LoRA なしで試す
- prompt は被写体説明だけでなく動きを書く

例:

`cinematic portrait of a woman slowly turning toward the camera, gentle blinking, subtle hair motion, natural lighting`

## LoRA について

デフォルトで workflow に入っている LoRA:

- `high_noise_model.safetensors`
- `low_noise_model.safetensors`

custom LoRA を使う場合:

- `/runpod-volume/loras` に置く
- `lora_pairs` でファイル名を指定する
- weight はまず `1.0`
- 強すぎる場合は `0.7` から `0.9` に下げる

## ローカル UI

このリポジトリには [`local_ui_server.py`](/Users/jimmy/Projects/generate_video/local_ui_server.py) を追加してあります。

起動方法:

```bash
python -m pip install flask requests
export RUNPOD_ENDPOINT_ID=your-endpoint-id
export RUNPOD_API_KEY=your-api-key
python local_ui_server.py
```

その後 `http://127.0.0.1:8787` を開いて使えます。

## RunPod で効率よく使うための推奨

- `/runpod-volume/models` をマウントする
- `/runpod-volume/loras` を使う
- 出力は可能なら `bucket_url` を使う
- 頻繁にモデルを切り替える場合は `refresh_worker` を検討する

## よくある誤解

- `gpu_profile` を変えれば同じ Endpoint の GPU が切り替わる
  切り替わりません。Endpoint 設定が優先です。
- GGUF を使うなら workflow JSON を全面的に作り直す必要がある
  このリポジトリでは handler がモデルファイル名と量子化設定を差し替えるので、基本は API 入力だけで使えます。
