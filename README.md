# Qwen3-TTS Voice-Design Service

A self-hosted, GPU-accelerated **Qwen3-TTS** REST API service built for
**audiobook production pipelines**.  It gives external clients full control
over voice design — narrators, characters, emotions, moods, and speech
styles — through a rich template system and free-form natural-language
voice descriptions.

Optimised for **English, German, and Portuguese**.  Runs as a native
**Ubuntu systemd service** on a single NVIDIA GPU.

---

## Table of Contents

- [Why Voice Design?](#why-voice-design)
- [Features](#features)
- [Architecture](#architecture)
- [Hardware requirements](#hardware-requirements)
- [Installation (Ubuntu)](#installation-ubuntu)
- [Managing the service](#managing-the-service)
- [Configuration](#configuration)
- [API reference](#api-reference)
  - [Health endpoints](#health-endpoints)
  - [OpenAI-compatible endpoint](#openai-compatible-endpoint)
  - [Voice design endpoint](#voice-design-endpoint)
  - [Voice templates](#voice-templates)
  - [Emotion & pace modifiers](#emotion--pace-modifiers)
  - [Custom voice endpoint (named speakers)](#custom-voice-endpoint-named-speakers)
  - [Voice clone endpoint](#voice-clone-endpoint)
  - [Batch synthesis](#batch-synthesis)
- [Voice template catalogue](#voice-template-catalogue)
- [Integration guide for audiobook pipelines](#integration-guide-for-audiobook-pipelines)
  - [Recommended workflow](#recommended-workflow)
  - [Choosing voices for characters](#choosing-voices-for-characters)
  - [Adapting to scene context](#adapting-to-scene-context)
  - [Sequential chapter processing](#sequential-chapter-processing)
  - [Error handling](#error-handling)
- [Example client](#example-client)
- [Switching models](#switching-models)
- [Supported languages](#supported-languages)
- [Performance & stability](#performance--stability)
- [Docker (alternative)](#docker-alternative)

---

## Why Voice Design?

The **VoiceDesign** model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`) is the default
because it is the only model variant that lets clients **create any voice they
can describe** — there is no fixed speaker roster.  For an audiobook pipeline
that must voice dozens of characters across every conceivable emotional
situation, this flexibility is essential.

| Capability | CustomVoice | **VoiceDesign** (default) | Base |
|---|---|---|---|
| Fixed named speakers | ✅ 9 speakers | ❌ | ❌ |
| Create unlimited voices from text | ❌ | **✅** | ❌ |
| Clone a real voice from audio | ❌ | ❌ | ✅ |
| Emotion/style control | partial | **full** | basic |
| Best for audiobooks? | good | **best** | situational |

---

## Features

| Capability | Detail |
|---|---|
| **Model** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` — 1.7 B parameters, ~3.4 GB VRAM in bfloat16 |
| **Languages** | English, German, Portuguese + Chinese, Japanese, Korean, French, Russian, Spanish, Italian |
| **Voice templates** | 50+ built-in presets across 6 categories (narrators, characters, emotions, moods, styles, languages) |
| **Composable voice design** | Combine a template + emotion modifier + pace modifier + free-form text |
| **API** | OpenAI-compatible `/v1/audio/speech` + extended `/v1/tts/*` endpoints |
| **Output formats** | WAV (lossless, default), FLAC, OGG, MP3 |
| **Batch inference** | Multiple texts in one request, returns base64 JSON array |
| **Sequential processing** | One request at a time — full GPU power for every generation |
| **Auth** | Optional bearer-token guard (`TTS_API_KEY`) |
| **Deployment** | Native Ubuntu systemd service (recommended) or Docker |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Audiobook Pipeline Client                              │
│  (sends chapter text + voice template + emotion + pace) │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP POST
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Qwen3-TTS Voice-Design Service (this repo)             │
│                                                          │
│  FastAPI + uvicorn (1 worker, 1 GPU)                     │
│  ├── /v1/audio/speech      (OpenAI-compatible)           │
│  ├── /v1/tts/voice-design  (template + modifiers)        │
│  ├── /v1/tts/voice-templates (browse presets)            │
│  └── /v1/tts/custom-voice  (named speakers)              │
│                                                          │
│  Threading lock → sequential GPU inference               │
│  CUDA cache trimmed after every generation               │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
              NVIDIA GPU (RTX 4070 12 GB)
              Qwen3-TTS-12Hz-1.7B-VoiceDesign
              bfloat16 + FlashAttention 2
```

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB) | **RTX 4070 (12 GB)** |
| CUDA | 11.8 | **12.1** |
| CPU | 4 cores | 10+ cores |
| RAM | 16 GB | **32 GB** |
| Disk | 10 GB | 20 GB |
| OS | Ubuntu 22.04 | **Ubuntu 24.04** |

---

## Installation (Ubuntu)

```bash
git clone https://github.com/stillquietlyloud/qwen3-tts.git
cd qwen3-tts

# Install drivers, CUDA, Python venv, deps, and register systemd service
sudo bash setup.sh
```

`setup.sh` does the following:

1. Installs system packages (build-essential, ffmpeg, Python 3.12, etc.)
2. Installs the NVIDIA driver if missing (reboot required on first run)
3. Installs CUDA Toolkit 12.1
4. Creates a Python 3.12 virtual environment at `.venv/`
5. Installs PyTorch 2.3 (CUDA 12.1), FlashAttention 2, and application deps
6. Creates `/etc/qwen3-tts.env` — the configuration file
7. Installs and enables `qwen3-tts.service` in systemd

On first start the model (~3.4 GB) is downloaded automatically from
HuggingFace.  To pre-download:

```bash
source .venv/bin/activate
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

---

## Managing the service

```bash
# Start / stop / restart
sudo systemctl start qwen3-tts
sudo systemctl stop qwen3-tts
sudo systemctl restart qwen3-tts

# View real-time logs
journalctl -u qwen3-tts -f

# Check status
systemctl status qwen3-tts

# Edit configuration
sudo nano /etc/qwen3-tts.env
sudo systemctl restart qwen3-tts
```

Or run manually for development:

```bash
source .venv/bin/activate
python -m server.main
```

---

## Configuration

All settings live in `/etc/qwen3-tts.env` (created by `setup.sh`).  They
can also be set as environment variables.

| Variable | Default | Description |
|---|---|---|
| `TTS_HOST` | `0.0.0.0` | Listen address |
| `TTS_PORT` | `8000` | Listen port |
| `TTS_DEFAULT_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | HuggingFace model ID |
| `TTS_MODEL_LOCAL_DIR` | _(empty)_ | Local model directory (overrides HF download) |
| `TTS_DEVICE` | `cuda:0` | PyTorch device |
| `TTS_DTYPE` | `bfloat16` | Tensor precision |
| `TTS_FLASH_ATTN` | `1` | Enable FlashAttention 2 (`0` to disable) |
| `TTS_DEFAULT_LANGUAGE` | `English` | Default language when not specified |
| `TTS_DEFAULT_VOICE_INSTRUCT` | _(warm male narrator)_ | Default voice instruct when no template is given |
| `TTS_OUTPUT_FORMAT` | `wav` | Default audio format |
| `TTS_MAX_NEW_TOKENS` | `4096` | Max tokens per generation |
| `TTS_API_KEY` | _(empty)_ | Bearer token; leave empty to disable auth |

---

## API reference

Interactive Swagger docs: **http://\<your-ip\>:8000/docs**

### Health endpoints

```
GET  /health          Liveness probe (always 200)
GET  /health/ready    Readiness probe (503 until model loaded)
GET  /v1/models       List loaded model (OpenAI-compatible)
```

---

### OpenAI-compatible endpoint

```
POST /v1/audio/speech
```

Drop-in replacement for the OpenAI TTS API.  When the VoiceDesign model is
loaded, the `voice` field accepts a **template ID** or a free-form voice
description instead of a speaker name.

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | **required** | Text to synthesise |
| `voice` | string | _(server default)_ | Template ID (e.g. `"narrator-omniscient"`) or free-form voice description |
| `model` | string | ignored | Ignored; uses loaded model |
| `response_format` | string | `wav` | `wav`, `mp3`, `ogg`, `flac` |
| `language` | string | `English` | Target language |
| `instruct` | string | _(none)_ | Extra style instruction layered on top |
| `template_id` | string | _(none)_ | Explicit template ID (takes priority over `voice`) |
| `emotion` | string | _(none)_ | Emotion modifier (e.g. `"joy"`, `"anger"`, `"fear"`) |
| `pace` | string | _(none)_ | Pace modifier (e.g. `"slow"`, `"fast"`, `"variable"`) |

**Example — using a template:**

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Chapter One. The adventure begins at dawn.",
    "voice": "narrator-omniscient",
    "language": "English",
    "emotion": "awe",
    "pace": "slow"
  }' --output chapter1.wav
```

**Example — free-form voice description:**

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Willkommen in der Welt der Geschichten.",
    "voice": "A warm, articulate German female narrator, mid-30s, clear Hochdeutsch.",
    "language": "German"
  }' --output german_intro.wav
```

---

### Voice design endpoint

```
POST /v1/tts/voice-design
```

The primary endpoint for audiobook pipelines.  Supports three ways to define
a voice — template, free-form, or a combination of both.

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string or string[] | **required** | Text(s) to synthesise |
| `language` | string or string[] | `English` | Target language(s) |
| `template_id` | string | _(none)_ | Voice template ID |
| `instruct` | string or string[] | _(none)_ | Free-form voice description or extra modifier |
| `emotion` | string | _(none)_ | Emotion modifier key |
| `pace` | string | _(none)_ | Pace modifier key |
| `response_format` | string | `wav` | Output format |
| `max_new_tokens` | int | 4096 | Max generation tokens |

**How the instruct is composed:**

```
[template instruct] + [emotion modifier] + [pace modifier] + [extra instruct]
```

The model receives this as a single combined prompt describing the voice.

**Example — template + emotion:**

```bash
curl -s http://localhost:8000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "You thought you could defeat me? How delightfully naive.",
    "language": "English",
    "template_id": "char-villain",
    "emotion": "contempt",
    "pace": "slow"
  }' --output villain.wav
```

**Example — pure free-form:**

```bash
curl -s http://localhost:8000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A vida é bela quando compartilhamos sonhos.",
    "language": "Portuguese",
    "instruct": "A vibrant Brazilian female voice, early 30s, warm mezzo-soprano, melodic and expressive."
  }' --output portuguese.wav
```

---

### Voice templates

```
GET /v1/tts/voice-templates              List all templates
GET /v1/tts/voice-templates?category=narrator   Filter by category
GET /v1/tts/voice-templates/{template_id}       Get single template
```

Each template contains a pre-written `instruct` optimised for the
Qwen3-TTS-VoiceDesign model.  Categories:

| Category | Templates | Purpose |
|---|---|---|
| `narrator` | 12 | Main storytelling voices (omniscient, first-person, thriller, fantasy, …) |
| `character` | 13 | Character archetypes (hero, villain, mentor, child, trickster, …) |
| `emotion` | 10 | Emotional delivery overlays (joy, sadness, anger, fear, awe, …) |
| `mood` | 8 | Scene atmosphere (suspense, romantic, action, eerie, triumph, …) |
| `style` | 8 | Speech styles (whisper, shout, inner thought, prayer, sarcasm, …) |
| `language` | 7 | Language-specific narrator voices (British, American, German, Portuguese variants) |

---

### Emotion & pace modifiers

```
GET /v1/tts/emotion-modifiers    List all emotion modifier keys
GET /v1/tts/pace-modifiers       List all pace modifier keys
```

**Emotion modifiers** (append to any template or free-form instruct):

| Key | Effect |
|---|---|
| `joy` | Bright, beaming, warm |
| `sadness` | Lower voice, slower, heavy with grief |
| `anger` | Clipped, forceful, sharp |
| `fear` | Trembling, rising pitch, nervous |
| `surprise` | Sharp pitch rise, breathless |
| `tenderness` | Soft, intimate, barely a whisper |
| `determination` | Firm, steady, building conviction |
| `despair` | Hollow, flat, drained |
| `awe` | Reverent near-whisper, humbled |
| `contempt` | Cold, mocking, dismissive |
| `urgency` | Fast, breathless, every second counts |
| `calm` | Even, unhurried, grounded |
| `excitement` | Quick pace, rising energy |
| `weariness` | Slow, heavy, effortful |
| `defiance` | Sharp, unyielding |
| `longing` | Slow, yearning |
| `playfulness` | Bouncy, teasing |
| `menace` | Low, soft, dangerously calm |
| `reverence` | Hushed, respectful |
| `bitterness` | Hard consonants, acid undertones |

**Pace modifiers:**

| Key | Effect |
|---|---|
| `very-slow` | Meditative, long pauses |
| `slow` | Deliberate, words land fully |
| `moderate` | Natural reading pace |
| `fast` | Energetic, driving forward |
| `very-fast` | Rapid-fire, breathless |
| `variable` | Dynamic, shifts with content |

---

### Custom voice endpoint (named speakers)

```
POST /v1/tts/custom-voice
```

Requires the **CustomVoice** model variant (`TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`).

Available speakers: `Ryan`, `Aiden` (English), `Vivian`, `Serena`, `Uncle_Fu`,
`Dylan`, `Eric` (Chinese), `Ono_Anna` (Japanese), `Sohee` (Korean).

```bash
curl -s http://localhost:8000/v1/tts/custom-voice \
  -H "Content-Type: application/json" \
  -d '{"text":"Guten Morgen.","language":"German","speaker":"Ryan"}' \
  --output german.wav
```

---

### Voice clone endpoint

```
POST /v1/tts/voice-clone
```

Requires the **Base** model (`TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base`).

```json
{
  "text": "This will sound like the reference speaker.",
  "language": "English",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Exact transcript of the reference audio."
}
```

---

### Batch synthesis

Send a list for `text` to synthesise multiple clips in one request:

```json
{
  "text": ["Chapter One.", "Chapter Two.", "Chapter Three."],
  "language": "English",
  "template_id": "narrator-omniscient"
}
```

Response:

```json
{
  "sample_rate": 24000,
  "format": "wav",
  "audio": ["<base64-clip-1>", "<base64-clip-2>", "<base64-clip-3>"]
}
```

---

## Voice template catalogue

### Narrators

| Template ID | Name | Best for |
|---|---|---|
| `narrator-omniscient` | Omniscient Narrator | Literary fiction, third-person |
| `narrator-first-person` | First-Person Narrator | Memoirs, confessional fiction |
| `narrator-female-literary` | Female Literary Narrator | Historical fiction, literary novels |
| `narrator-documentary` | Documentary Narrator | Non-fiction, biographies |
| `narrator-bedtime` | Bedtime Story Narrator | Children's books |
| `narrator-thriller` | Thriller Narrator | Crime, suspense, mystery |
| `narrator-epic-fantasy` | Epic Fantasy Narrator | Fantasy, mythology, epic sagas |
| `narrator-humorous` | Humorous Narrator | Comedy, satire, light fiction |
| `narrator-noir` | Noir Narrator | Detective fiction, hardboiled crime |
| `narrator-romance` | Romance Narrator | Romance novels |
| `narrator-horror` | Horror Narrator | Horror, dark fiction |
| `narrator-young-adult` | Young Adult Narrator | YA fiction |

### Characters

| Template ID | Name | Archetype |
|---|---|---|
| `char-hero` | Hero | Brave, noble, determined |
| `char-heroine` | Heroine | Fierce, determined, empathetic |
| `char-villain` | Villain | Calculating, menacing, charming evil |
| `char-villainess` | Villainess | Cunning, elegant, ruthless |
| `char-mentor` | Wise Mentor | Patient, knowledgeable, fatherly |
| `char-child-boy` | Young Boy | Innocent, curious, energetic |
| `char-child-girl` | Young Girl | Sweet, imaginative, lively |
| `char-trickster` | Trickster | Cunning, playful, unpredictable |
| `char-warrior` | Battle-Hardened Warrior | Tough, blunt, weathered |
| `char-royal` | Royal / Noble | Refined, formal, commanding |
| `char-mystic` | Mystic / Oracle | Ethereal, prophetic, otherworldly |
| `char-companion` | Loyal Companion | Warm, reliable, supportive |
| `char-femme-fatale` | Femme Fatale | Sultry, dangerous, seductive |
| `char-comic-relief` | Comic Relief | Bumbling, excitable, lovable |
| `char-elder-woman` | Elder Woman | Wise, gentle, resilient |

### Emotions

| Template ID | Name | Delivery |
|---|---|---|
| `emotion-joy` | Joyful | Bright, warm, fast |
| `emotion-sadness` | Sorrowful | Slow, heavy, trembling |
| `emotion-anger` | Angry | Tight, forceful, sharp |
| `emotion-fear` | Frightened | Shaky, breathless, high |
| `emotion-surprise` | Surprised | Sharp pitch rise |
| `emotion-tenderness` | Tender | Soft, intimate, slow |
| `emotion-determination` | Determined | Firm, steady, rising |
| `emotion-despair` | Despairing | Hollow, flat, broken |
| `emotion-awe` | Awestruck | Hushed, reverent, slow |
| `emotion-contempt` | Contemptuous | Cold, mocking |

### Moods

| Template ID | Name | Atmosphere |
|---|---|---|
| `mood-suspense` | Suspenseful | Tense, hushed, slow |
| `mood-romantic` | Romantic | Soft, dreamy, intimate |
| `mood-action` | Action / Combat | Fast, punchy, urgent |
| `mood-melancholy` | Melancholy | Wistful, reflective |
| `mood-triumph` | Triumphant | Powerful, rising, celebratory |
| `mood-eerie` | Eerie / Uncanny | Hollow, unsettling |
| `mood-peaceful` | Peaceful | Gentle, serene, calm |
| `mood-epic-battle` | Epic Battle | Grand, intense |

### Styles

| Template ID | Name | Use case |
|---|---|---|
| `style-whisper` | Whisper | Secrets, danger |
| `style-shout` | Shout / Command | Battle cries, commands |
| `style-inner-thought` | Inner Thought | Internal monologue |
| `style-letter-reading` | Reading a Letter | Correspondence |
| `style-prayer` | Prayer / Incantation | Ritual, sacred scenes |
| `style-proclamation` | Royal Proclamation | Decrees, speeches |
| `style-sarcastic` | Sarcastic | Dry humour |
| `style-dialogue-excited` | Excited Dialogue | Enthusiastic exchanges |

### Language-specific

| Template ID | Name | Language |
|---|---|---|
| `lang-en-british-narrator` | British English Narrator | English (RP) |
| `lang-en-american-narrator` | American English Narrator | English (GenAm) |
| `lang-de-narrator` | German Narrator (Hochdeutsch) | German |
| `lang-de-female-narrator` | German Female Narrator | German |
| `lang-pt-br-narrator` | Brazilian Portuguese Narrator | Portuguese (BR) |
| `lang-pt-eu-narrator` | European Portuguese Narrator | Portuguese (EU) |
| `lang-pt-female-narrator` | Portuguese Female Narrator | Portuguese |

---

## Integration guide for audiobook pipelines

### Recommended workflow

```
1. Client parses book → list of segments (narration, dialogue, …)
2. For each segment:
   a. Choose a voice template (or reuse a consistent character voice)
   b. Set emotion based on scene context
   c. Set pace based on action level
   d. POST /v1/tts/voice-design
   e. Save the returned WAV
   f. Wait for response before sending the next request
3. Concatenate WAV files into final audiobook
```

### Choosing voices for characters

Create a **character voice map** at the start of your pipeline:

```python
VOICES = {
    "narrator":   {"template_id": "narrator-omniscient"},
    "protagonist": {"template_id": "char-hero"},
    "antagonist":  {"template_id": "char-villain"},
    "mentor":      {"template_id": "char-mentor"},
    "love_interest": {"template_id": "char-heroine"},
    "comic_sidekick": {"template_id": "char-comic-relief"},
    "child":       {"template_id": "char-child-boy"},
    # Custom voice for a specific minor character:
    "innkeeper": {
        "instruct": "A jolly, round-bellied male voice, 50s, deep and "
                    "booming with a hearty laugh always lurking. Speaks "
                    "with a thick regional accent and infectious warmth."
    },
}
```

Then for each segment:

```python
voice = VOICES[segment.character]
response = client.post("/v1/tts/voice-design", json={
    "text": segment.text,
    "language": segment.language,
    **voice,
    "emotion": segment.detected_emotion,  # from your NLP pipeline
    "pace": "fast" if segment.is_action else "moderate",
    "response_format": "wav",
})
```

### Adapting to scene context

| Scene type | Recommended approach |
|---|---|
| **Narration (calm)** | `narrator-omniscient` + `pace: "moderate"` |
| **Narration (tense)** | `narrator-thriller` + `emotion: "urgency"` |
| **Dialogue (argument)** | Character template + `emotion: "anger"` |
| **Dialogue (confession)** | Character template + `emotion: "tenderness"` + `pace: "slow"` |
| **Action sequence** | `mood-action` or `mood-epic-battle` + `pace: "fast"` |
| **Horror scene** | `narrator-horror` + `emotion: "fear"` |
| **Romantic moment** | `mood-romantic` + `emotion: "tenderness"` + `pace: "slow"` |
| **Comedic beat** | `narrator-humorous` + `emotion: "playfulness"` |
| **Inner monologue** | `style-inner-thought` + character emotion |
| **Letter / diary** | `style-letter-reading` |
| **Spell / prayer** | `style-prayer` + `emotion: "reverence"` |
| **Royal decree** | `style-proclamation` |
| **Whispered secret** | `style-whisper` + `emotion: "fear"` or `"menace"` |
| **Battle cry** | `style-shout` + `emotion: "determination"` |
| **Chapter opening** | Narrator template + `emotion: "awe"` + `pace: "slow"` |
| **Cliffhanger ending** | `mood-suspense` + `pace: "slow"` |
| **Triumphant finale** | `mood-triumph` + `emotion: "joy"` |
| **Tragic ending** | Narrator template + `emotion: "despair"` + `pace: "very-slow"` |
| **German narration** | `lang-de-narrator` or `lang-de-female-narrator` |
| **Portuguese narration** | `lang-pt-br-narrator` or `lang-pt-eu-narrator` |

### Sequential chapter processing

The service processes **one request at a time** (threading lock on the GPU).
For long audiobook pipelines:

```python
import httpx

BASE_URL = "http://your-server:8000"

with httpx.Client(base_url=BASE_URL, timeout=300) as client:
    for i, segment in enumerate(book_segments):
        response = client.post("/v1/tts/voice-design", json={
            "text": segment.text,
            "language": segment.language,
            "template_id": segment.voice_template,
            "emotion": segment.emotion,
            "pace": segment.pace,
            "response_format": "wav",
        })
        response.raise_for_status()

        with open(f"output/segment_{i:05d}.wav", "wb") as f:
            f.write(response.content)
```

**Tips:**
- Set `timeout=300` (5 minutes) for long text segments
- The server frees CUDA memory after each generation, so memory usage stays stable
- WAV format is recommended for lossless quality; convert to MP3 client-side if needed

### Error handling

| HTTP status | Meaning | Action |
|---|---|---|
| 200 | Success | Save the audio |
| 422 | Invalid request (bad template ID, wrong model) | Fix the request |
| 500 | Inference error | Retry; check server logs |
| 503 | Model not loaded yet | Wait and retry |

---

## Example client

```bash
source .venv/bin/activate
pip install httpx

# Quick mode — browse templates only (no audio generation)
python examples/example_client.py --base-url http://localhost:8000 --quick

# Full mode — generate all example audio files
python examples/example_client.py --base-url http://localhost:8000
```

The script demonstrates:
- Browsing all templates, emotions, and pace modifiers
- Using the OpenAI-compatible endpoint with a template ID
- Voice design with template + emotion + pace combinations
- Multi-language narration (English, German, Portuguese)
- Character dialogue scene with voice switching
- Fully custom free-form voice instruct
- Mood comparison (same text, different atmospheres)

---

## Switching models

Edit `/etc/qwen3-tts.env` and restart the service:

| Use case | Model | Set in env |
|---|---|---|
| **Voice design (recommended)** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Named speakers | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Voice cloning | `Qwen3-TTS-12Hz-1.7B-Base` | `TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| Low VRAM / faster | `Qwen3-TTS-12Hz-0.6B-VoiceDesign` | `TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign` |

---

## Supported languages

| Language | Code | Quality |
|---|---|---|
| English | `English` | ★★★★★ Native |
| German | `German` | ★★★★★ Native |
| Portuguese | `Portuguese` | ★★★★★ Native |
| Chinese | `Chinese` | ★★★★★ Native |
| Japanese | `Japanese` | ★★★★★ Native |
| Korean | `Korean` | ★★★★★ Native |
| French | `French` | ★★★★☆ |
| Russian | `Russian` | ★★★★☆ |
| Spanish | `Spanish` | ★★★★☆ |
| Italian | `Italian` | ★★★★☆ |

Voice description prompts (`instruct`) must be written in **English** or
**Chinese**, regardless of the output language.

---

## Performance & stability

| Tip | Impact |
|---|---|
| Use `bfloat16` (default) with FlashAttention 2 | ~30 % faster, ~25 % less VRAM |
| One worker, one GPU (enforced) | Maximum throughput per request |
| CUDA cache cleared after every generation | Stable VRAM over 1000s of requests |
| systemd `Restart=on-failure` | Auto-recovery from crashes |
| Set `TTS_MAX_NEW_TOKENS` to match longest chapter | Prevents truncation |
| Use WAV output; convert to MP3 client-side | Saves server CPU |

---

## Docker (alternative)

Docker Compose is available as an alternative deployment method:

```bash
docker compose up -d
docker compose logs -f
```

For production audiobook pipelines the **native systemd service** is
recommended — it avoids the container overhead and gives the service direct
GPU access for maximum performance.
