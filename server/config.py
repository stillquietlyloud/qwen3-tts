"""
Configuration for the Qwen3-TTS API server.
All settings can be overridden via environment variables.
"""

import os
from typing import Optional


# ── Server ──────────────────────────────────────────────────────────────────
HOST: str = os.getenv("TTS_HOST", "0.0.0.0")
PORT: int = int(os.getenv("TTS_PORT", "8000"))
WORKERS: int = int(os.getenv("TTS_WORKERS", "1"))  # keep 1: single GPU

# ── Model selection ──────────────────────────────────────────────────────────
# VoiceDesign 1.7B is the default: it allows clients to describe any voice
# they need via natural-language instructions, which is essential for an
# audiobook pipeline that must adapt voices to every situation.
# The 1.7B model fits comfortably on a 12 GB RTX 4070 (~3.4 GB bfloat16).
DEFAULT_MODEL: str = os.getenv(
    "TTS_DEFAULT_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)

# Alternative models (set via env var):
#   TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  # 9 named speakers
#   TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base          # voice cloning

# Path to a local model directory (leave empty to auto-download from HF/MS)
MODEL_LOCAL_DIR: Optional[str] = os.getenv("TTS_MODEL_LOCAL_DIR") or None

# ── GPU / precision ──────────────────────────────────────────────────────────
DEVICE: str = os.getenv("TTS_DEVICE", "cuda:0")
# bfloat16 is required for FlashAttention 2 and gives the best quality/speed.
DTYPE: str = os.getenv("TTS_DTYPE", "bfloat16")
USE_FLASH_ATTN: bool = os.getenv("TTS_FLASH_ATTN", "1") not in ("0", "false", "False")

# ── Audio output ─────────────────────────────────────────────────────────────
# WAV is lossless and preferred for audiobooks.
# Accepted values: "wav", "mp3", "ogg", "flac"
DEFAULT_OUTPUT_FORMAT: str = os.getenv("TTS_OUTPUT_FORMAT", "wav")

# ── Generation defaults ──────────────────────────────────────────────────────
DEFAULT_LANGUAGE: str = os.getenv("TTS_DEFAULT_LANGUAGE", "English")
DEFAULT_SPEAKER: str = os.getenv("TTS_DEFAULT_SPEAKER", "Ryan")

# Default voice-design instruct used when no instruct or template is given.
DEFAULT_VOICE_INSTRUCT: str = os.getenv(
    "TTS_DEFAULT_VOICE_INSTRUCT",
    "A warm, clear male narrator voice, mid-30s, natural midrange, "
    "measured pace, suitable for audiobook narration.",
)

# Max tokens the model may generate per request
MAX_NEW_TOKENS: int = int(os.getenv("TTS_MAX_NEW_TOKENS", "4096"))

# ── Cache / temp dirs ────────────────────────────────────────────────────────
CACHE_DIR: str = os.getenv("TTS_CACHE_DIR", "/tmp/tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── API key (optional, simple bearer-token guard for LAN use) ────────────────
API_KEY: Optional[str] = os.getenv("TTS_API_KEY") or None
