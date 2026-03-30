"""
Pydantic request / response models for the Qwen3-TTS API.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Shared enums ─────────────────────────────────────────────────────────────

class AudioFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    ogg = "ogg"
    flac = "flac"


class Language(str, Enum):
    auto = "Auto"
    chinese = "Chinese"
    english = "English"
    japanese = "Japanese"
    korean = "Korean"
    german = "German"
    french = "French"
    russian = "Russian"
    portuguese = "Portuguese"
    spanish = "Spanish"
    italian = "Italian"


# Speakers available in CustomVoice models
CUSTOM_VOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


# ── OpenAI-compatible /v1/audio/speech ───────────────────────────────────────

class OpenAISpeechRequest(BaseModel):
    """Compatible with the OpenAI Audio Speech API."""

    model: str = Field(
        default="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        description="TTS model identifier (ignored; the server uses whatever model is loaded).",
    )
    input: str = Field(..., description="Text to synthesise.")
    voice: str = Field(
        default="",
        description=(
            "For VoiceDesign: a natural-language voice description, or a "
            "voice-template ID (e.g. 'narrator-omniscient'). "
            "For CustomVoice: a speaker name (e.g. 'Ryan')."
        ),
    )
    response_format: AudioFormat = Field(
        default=AudioFormat.wav,
        description="Output audio format.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier (not yet honoured by the model; reserved for future use).",
    )
    # Extensions (not in the OpenAI spec)
    language: Optional[Language] = Field(
        default=None,
        description="Target language.  Defaults to the server's DEFAULT_LANGUAGE.",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Natural-language style instruction, e.g. 'Speak in a warm, soothing tone.'",
    )
    template_id: Optional[str] = Field(
        default=None,
        description=(
            "Voice template ID (e.g. 'narrator-omniscient', 'char-villain'). "
            "When set, the template's instruct is used as the base voice, and "
            "'instruct' becomes an additional modifier layered on top."
        ),
    )
    emotion: Optional[str] = Field(
        default=None,
        description=(
            "Emotion modifier key (e.g. 'joy', 'anger', 'fear'). "
            "Appended to the instruct to colour the delivery."
        ),
    )
    pace: Optional[str] = Field(
        default=None,
        description=(
            "Pace modifier key (e.g. 'slow', 'fast', 'variable'). "
            "Appended to the instruct to control delivery speed."
        ),
    )


# ── Custom voice ──────────────────────────────────────────────────────────────

class CustomVoiceRequest(BaseModel):
    text: str | List[str] = Field(..., description="Text(s) to synthesise.")
    language: Language | List[Language] = Field(
        default=Language.english,
        description="Target language(s).",
    )
    speaker: str | List[str] = Field(
        default="Ryan",
        description="Speaker name(s) from the supported speaker list.",
    )
    instruct: Optional[str | List[str]] = Field(
        default=None,
        description="Optional style instruction(s).",
    )
    response_format: AudioFormat = Field(default=AudioFormat.wav)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)


# ── Voice design ──────────────────────────────────────────────────────────────

class VoiceDesignRequest(BaseModel):
    text: str | List[str] = Field(..., description="Text(s) to synthesise.")
    language: Language | List[Language] = Field(default=Language.english)
    instruct: Optional[str | List[str]] = Field(
        default=None,
        description=(
            "Natural-language description of the desired voice. "
            "If template_id is also provided, this is appended as an extra modifier."
        ),
    )
    template_id: Optional[str] = Field(
        default=None,
        description=(
            "Voice template ID (e.g. 'narrator-omniscient', 'char-villain'). "
            "Use GET /v1/tts/voice-templates to list all available templates."
        ),
    )
    emotion: Optional[str] = Field(
        default=None,
        description="Emotion modifier key (e.g. 'joy', 'anger', 'fear', 'tenderness').",
    )
    pace: Optional[str] = Field(
        default=None,
        description="Pace modifier key (e.g. 'slow', 'fast', 'variable').",
    )
    response_format: AudioFormat = Field(default=AudioFormat.wav)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)


# ── Voice clone ───────────────────────────────────────────────────────────────

class VoiceCloneRequest(BaseModel):
    text: str | List[str] = Field(..., description="Text(s) to synthesise.")
    language: Language | List[Language] = Field(default=Language.english)
    ref_audio: str = Field(
        ...,
        description=(
            "Reference audio: local file path, public URL, or base64-encoded "
            "PCM/WAV string."
        ),
    )
    ref_text: Optional[str] = Field(
        default=None,
        description=(
            "Transcript of the reference audio.  Recommended for best quality; "
            "omit to use x-vector-only mode."
        ),
    )
    response_format: AudioFormat = Field(default=AudioFormat.wav)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)


# ── Shared response ───────────────────────────────────────────────────────────

class TTSErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
