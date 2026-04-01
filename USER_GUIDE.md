# USER GUIDE

## これは何か

このリポジトリは、Wan 2.2 を使って画像から動画を生成する `runpod.io` Serverless Worker です。内部では ComfyUI を起動し、RunPod の `/run` API からジョブを受け取ります。重いモデル本体は Docker に焼かず、必要時に `/runpod-volume` へ取得する構成です。

## できること

- 通常の Wan 2.2 FP8 モデルを使う
- Wan 2.2 の GGUF 量子化モデルを使う
- `model_profile` を明示してモデルを切り替える
- 必要なら `target_vram_gb` からモデル選択を補助する
- 出力を `base64` か `bucket_url` で返す
- RunPod のリクエスト上位 `s3Config` を使ったアップロードもできる

## 重要な前提

RunPod Serverless の実機 GPU は Endpoint 設定で決まります。リクエスト body で「この GPU を使う」はできません。

そのため、このリポジトリでも基本は `model_profile` を明示するか、Endpoint 側の `DEFAULT_MODEL_PROFILE` を使うのが自然です。`gpu_profile` は後方互換のため残っていますが、新規利用では非推奨です。

## 主な入力パラメータ

- `prompt`
- `image_path` / `image_url` / `image_base64`
- `end_image_path` / `end_image_url` / `end_image_base64`
- `model_profile`
- `loras`
- `output_mode`
- `refresh_worker`
- `target_vram_gb`
- `civitai_token` / `huggingface_token`

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

これらは最初の利用時に自動取得され、その後は volume から再利用されます。

custom LoRA を使う場合:

- `/runpod-volume/loras` に置いて `loras[].source` にファイル名を書く
- または `loras[].source` に直接ダウンロードURLを書く
- または Civitai の model page URL を `modelVersionId` 付きで渡す
- または Hugging Face の `/resolve/` か `/blob/` のファイルURLを渡す
- gated/private な配布物は `civitai_token` または `huggingface_token` を併用する
- weight はまず `0.7` から `1.0`
- 強すぎる場合は `0.7` から `0.9` に下げる

例:

```json
{
  "input": {
    "prompt": "cinematic portrait of a woman turning toward the camera",
    "image_url": "https://example.com/input.png",
    "model_profile": "gguf_q4_k_m",
    "loras": [
      {
        "source": "my_style_lora.safetensors",
        "weight": 0.8
      },
      {
        "source": "https://civitai.com/models/122359/detail-tweaker-xl?modelVersionId=135867",
        "weight": 0.8
      },
      {
        "source": "https://huggingface.co/owner/repo/resolve/main/loras/my_style.safetensors",
        "weight": 0.8
      }
    ]
  }
}
```

## ローカル UI

このリポジトリには [`local_ui_server.py`](/Users/jimmy/Projects/generate_video/local_ui_server.py) を追加してあります。

起動方法:

```bash
uv sync
export RUNPOD_ENDPOINT_ID=your-endpoint-id
export RUNPOD_API_KEY=your-api-key
uv run python local_ui_server.py
```

その後 `http://127.0.0.1:8787` を開いて使えます。

UI には次のガイドを出しています。

- 24GB / 32GB / 40GB / 48GB+ ごとの推奨 `model_profile`
- Civitai / Hugging Face / 事前配置ファイル名の LoRA 入力例
- gated/private asset 用の Civitai token / Hugging Face token 入力欄

UI に入力した設定はブラウザの `localStorage` に保存されるため、同じブラウザならタブを閉じても復元されます。

- 自動保存されるもの: endpoint id、prompt、解像度、steps、LoRA 設定など
- 明示チェック時のみ保存されるもの: API key と token 類
- 保存されないもの: 画像ファイル本体

`localStorage` は便利ですが、暗号化された秘密保管庫ではありません。同じブラウザプロファイル上でそのページの JavaScript から読めます。したがって、共有PCや、信用できないスクリプトを同一 origin で動かす構成では API key を保存しないでください。ローカル専用UIを自分の端末で使う範囲なら、endpoint id や一般設定の保存は現実的です。

Python 依存関係は [`pyproject.toml`](/Users/jimmy/Projects/generate_video/pyproject.toml) で管理します。`local_ui_server.py` だけでなく、`generate_video_client.py` と `handler.py` の実行依存もここに寄せています。

## RunPod で効率よく使うための推奨

- `/runpod-volume/models` をマウントする
- `/runpod-volume/loras` を使う
- 出力は可能なら `bucket_url` を使う
- 頻繁にモデルを切り替える場合は `refresh_worker` を検討する
- `/runpod-volume` が満杯なら、lazy download は worker ローカルの ComfyUI model ディレクトリへフォールバックします

## RunPod Serverless 仕様との整合

この worker は、RunPod Serverless の標準 handler 方式に沿っています。

- `runpod.serverless.start({"handler": handler})` で起動する
- `refresh_worker` は RunPod が解釈できる返却形式で返す
- progress update は補助的に送る
- network volume は `/runpod-volume` 前提で使う

注意点:

- RunPod のレスポンス payload 制限に対して、動画を `base64` で返すのは危険です。このため現在は inline 出力サイズを保守的に制限しています。
- そのため、実運用では `output_mode="bucket_url"` を推奨します。
- `bucket_url` は endpoint の環境変数だけでなく、RunPod リクエスト上位の `s3Config` でも使えます。
- この worker は `/run` と `/runsync` を自動判別してしきい値を変えていません。安全側に倒した共通上限で判定します。

## よくある誤解

- `gpu_profile` を変えれば同じ Endpoint の GPU が切り替わる
  切り替わりません。Endpoint 設定が優先です。UIからも外しています。
- GGUF を使うなら workflow JSON を全面的に作り直す必要がある
  このリポジトリでは handler がモデルファイル名と量子化設定を差し替えるので、基本は API 入力だけで使えます。
