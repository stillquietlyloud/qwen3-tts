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
# Recommended for audiobooks: 1.7B-CustomVoice gives the highest-quality
# multilingual (EN/DE/PT) output while fitting in a 12 GB RTX 4070.
DEFAULT_MODEL: str = os.getenv(
    "TTS_DEFAULT_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)

# Uncomment or set env var to use VoiceDesign or Base instead:
#   TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
#   TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base

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

# Max tokens the model may generate per request
MAX_NEW_TOKENS: int = int(os.getenv("TTS_MAX_NEW_TOKENS", "4096"))

# ── Cache / temp dirs ────────────────────────────────────────────────────────
CACHE_DIR: str = os.getenv("TTS_CACHE_DIR", "/tmp/tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── API key (optional, simple bearer-token guard for LAN use) ────────────────
API_KEY: Optional[str] = os.getenv("TTS_API_KEY") or None
