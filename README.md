# qwen3-tts

A self-hosted, GPU-accelerated **Qwen3-TTS** REST API server for generating
high-quality speech from text.  Built for audiobook production, optimised for
English, German, and Portuguese, and designed to run on a single NVIDIA RTX 4070
(12 GB VRAM) on Ubuntu 24.04.

---

## Table of Contents

- [Features](#features)
- [Architecture decision](#architecture-decision)
- [Hardware requirements](#hardware-requirements)
- [Quick start – Docker (recommended)](#quick-start--docker-recommended)
- [Quick start – Bare metal](#quick-start--bare-metal)
- [Configuration](#configuration)
- [API reference](#api-reference)
  - [Health endpoints](#health-endpoints)
  - [OpenAI-compatible endpoint](#openai-compatible-endpoint)
  - [Extended TTS endpoints](#extended-tts-endpoints)
- [Example client](#example-client)
- [Switching models](#switching-models)
- [Language and speaker notes](#language-and-speaker-notes)
- [Performance tips](#performance-tips)

---

## Features

| Capability | Detail |
|---|---|
| **Languages** | English, German, Portuguese (+ Chinese, Japanese, Korean, French, Russian, Spanish, Italian) |
| **Synthesis modes** | Custom-voice (9 named speakers), Voice-design (describe in words), Voice-clone (3-second reference clip) |
| **API** | OpenAI-compatible `/v1/audio/speech` + extended `/v1/tts/*` endpoints |
| **Output formats** | WAV (lossless, default), FLAC, OGG, MP3 |
| **Batch inference** | Multiple texts in one request, returns base64 JSON array |
| **Auth** | Optional bearer-token guard (`TTS_API_KEY`) |
| **Deployment** | Docker Compose or systemd bare-metal service |

---

## Architecture decision

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **qwen-tts Python package** (chosen) | Simplest install, full feature set, streaming-ready, well-maintained | Slightly more CPU overhead than vLLM | ✅ **Recommended** |
| vLLM-Omni | Faster batch throughput for large queues | Online serving not yet supported; offline only | ❌ Excluded (no HTTP serving) |
| llama.cpp | Lowest RAM, runs on CPU | No Qwen3-TTS GGUF available; audio quality unknown | ❌ Excluded |

The **`qwen-tts`** library with the **`Qwen3-TTS-12Hz-1.7B-CustomVoice`** model
is the optimal choice:

- Fits comfortably in 12 GB VRAM (~3.4 GB in bfloat16)
- Supports all 10 languages at the highest quality tier
- Named speakers (Ryan, Aiden) are natively English and work great for audiobooks
- One model handles everything — no need to load two models

---

## Hardware requirements

| Component | Minimum | Recommended (this setup) |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB) | **RTX 4070 (12 GB)** |
| CUDA | 11.8 | **12.1** |
| CPU | 4 cores | **10 cores** |
| RAM | 16 GB | **32 GB** |
| Disk | 10 GB | 20 GB (model cache + Docker layers) |
| OS | Ubuntu 22.04 | **Ubuntu 24.04** |

---

## Quick start – Docker (recommended)

### Prerequisites

```bash
# 1. Docker Engine + Compose plugin
curl -fsSL https://get.docker.com | bash

# 2. NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Start the server

```bash
git clone https://github.com/stillquietlyloud/qwen3-tts.git
cd qwen3-tts

docker compose up -d          # builds image and starts container
docker compose logs -f        # watch startup / model download progress
```

The container downloads the model on first start (~3.4 GB). Once you see
`Model ready` in the logs the API is ready at **http://\<your-ip\>:8000**.

---

## Quick start – Bare metal

```bash
git clone https://github.com/stillquietlyloud/qwen3-tts.git
cd qwen3-tts

sudo bash setup.sh            # installs NVIDIA driver, CUDA, Python venv, systemd service
source .venv/bin/activate
python -m server.main         # starts the server
```

> **Note**: if the NVIDIA driver was just installed, `setup.sh` will ask you to
> reboot before re-running it.

---

## Configuration

All options are set via environment variables (or in `docker-compose.yml`).

| Variable | Default | Description |
|---|---|---|
| `TTS_HOST` | `0.0.0.0` | Listen address |
| `TTS_PORT` | `8000` | Listen port |
| `TTS_DEFAULT_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HuggingFace model ID |
| `TTS_MODEL_LOCAL_DIR` | _(empty)_ | Override with a local model directory path |
| `TTS_DEVICE` | `cuda:0` | PyTorch device |
| `TTS_DTYPE` | `bfloat16` | Tensor precision |
| `TTS_FLASH_ATTN` | `1` | Enable FlashAttention 2 (set `0` to disable) |
| `TTS_DEFAULT_LANGUAGE` | `English` | Default language if not specified in request |
| `TTS_DEFAULT_SPEAKER` | `Ryan` | Default speaker for CustomVoice requests |
| `TTS_OUTPUT_FORMAT` | `wav` | Default audio output format |
| `TTS_MAX_NEW_TOKENS` | `4096` | Max tokens per generation |
| `TTS_API_KEY` | _(empty)_ | Bearer token; leave empty to disable auth |
| `HF_HOME` | _(system default)_ | HuggingFace cache directory |

---

## API reference

Interactive docs available at **http://\<your-ip\>:8000/docs** once the server is running.

### Health endpoints

```
GET  /health          Liveness probe  (always 200)
GET  /health/ready    Readiness probe (503 until model is loaded)
GET  /v1/models       List loaded model (OpenAI-compatible)
```

### OpenAI-compatible endpoint

```
POST /v1/audio/speech
```

Drop-in replacement for the OpenAI TTS API.  Works with any OpenAI client
library — just point `base_url` at this server.

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | **required** | Text to synthesise |
| `voice` | string | `Ryan` | Speaker name (CustomVoice) or voice description (VoiceDesign) |
| `model` | string | ignored | Ignored; server uses whatever model is loaded |
| `response_format` | string | `wav` | `wav`, `mp3`, `ogg`, or `flac` |
| `language` | string | `English` | Target language |
| `instruct` | string | _(none)_ | Style instruction, e.g. `"Speak slowly and clearly."` |

**Example**:

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Chapter One. The adventure begins.",
    "voice": "Ryan",
    "language": "English",
    "instruct": "Warm audiobook narrator tone.",
    "response_format": "wav"
  }' \
  --output chapter1.wav
```

### Extended TTS endpoints

#### `POST /v1/tts/custom-voice`

Requires the **CustomVoice** model.

```json
{
  "text": "Your text here.",
  "language": "English",
  "speaker": "Ryan",
  "instruct": "Optional style instruction.",
  "response_format": "wav"
}
```

Supported speakers: `Ryan`, `Aiden` (English), `Vivian`, `Serena`, `Uncle_Fu`,
`Dylan`, `Eric` (Chinese), `Ono_Anna` (Japanese), `Sohee` (Korean).

```bash
curl -s http://localhost:8000/v1/tts/custom-voice \
  -H "Content-Type: application/json" \
  -d '{"text":"Guten Morgen, wie geht es Ihnen?","language":"German","speaker":"Ryan"}' \
  --output german.wav
```

#### `POST /v1/tts/voice-design`

Requires the **VoiceDesign** model (`TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`).

```json
{
  "text": "Once upon a time in a land far away…",
  "language": "English",
  "instruct": "Deep, wise, elderly male narrator voice with a slow, measured pace.",
  "response_format": "wav"
}
```

#### `POST /v1/tts/voice-clone`

Requires the **Base** model (`TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base`).

```json
{
  "text": "This will sound just like the reference speaker.",
  "language": "English",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Exact transcript of the reference audio clip.",
  "response_format": "wav"
}
```

`ref_audio` can be a local file path, a public URL, or a base64-encoded WAV string.

#### Batch synthesis

Send a list for `text` (and corresponding lists for `language`, `speaker`, etc.)
to synthesise multiple clips in one request.  The response is JSON:

```json
{
  "sample_rate": 24000,
  "format": "wav",
  "audio": ["<base64-clip-1>", "<base64-clip-2>", "..."]
}
```

#### `GET /v1/tts/speakers`

Returns the list of available speakers and their descriptions.

#### `GET /v1/tts/languages`

Returns the list of supported languages.

---

## Example client

```bash
pip install httpx
python examples/example_client.py --base-url http://<server-ip>:8000
```

The script runs through all endpoints and saves the generated WAV files to `/tmp/`.

---

## Switching models

Edit `docker-compose.yml` (or set the environment variable) to switch the
loaded model:

| Use case | Model |
|---|---|
| Audiobooks with named speakers (recommended) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Fully custom voice persona | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Clone a speaker from a recording | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| Lower VRAM / faster inference | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |

---

## Language and speaker notes

For **English audiobooks** use `Ryan` or `Aiden` — both are native English speakers
and produce the most natural prosody.

For **German** and **Portuguese** any speaker works well; `Ryan` and `Aiden` are
recommended as their English-based training still generalises cleanly to these
languages.

---

## Performance tips

| Tip | Impact |
|---|---|
| Use `bfloat16` (default) with FlashAttention 2 | ~30% faster, ~25% less VRAM |
| Keep batch size ≤ 8 on a 12 GB GPU | Avoids OOM on long texts |
| Mount model cache volume in Docker | Avoids re-downloading on container restart |
| Use WAV output locally; convert to MP3 client-side | Saves CPU on the server |
| Set `TTS_MAX_NEW_TOKENS` to match your longest chapter | Prevents truncation |
