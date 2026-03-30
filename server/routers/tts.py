"""
Extended TTS endpoints that expose the full Qwen3-TTS feature set.

POST /v1/tts/custom-voice   – custom-speaker synthesis
POST /v1/tts/voice-design   – voice designed from a text description
POST /v1/tts/voice-clone    – 3-second voice clone from a reference clip
GET  /v1/tts/speakers       – list available speakers
GET  /v1/tts/languages      – list supported languages

All endpoints return raw audio bytes when the request contains a single
text string, or a JSON list of base64-encoded audio clips for batch
requests.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .. import config
from ..model_manager import manager
from ..schemas import (
    CustomVoiceRequest,
    VoiceCloneRequest,
    VoiceDesignRequest,
)
from .auth import verify_api_key

logger = logging.getLogger("tts.tts")
router = APIRouter(prefix="/v1/tts", tags=["tts"], dependencies=[Depends(verify_api_key)])

_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


def _respond(wavs, sr: int, fmt: str):
    """Return a single audio Response or a JSON list of base64 clips."""
    if len(wavs) == 1:
        audio = manager.encode_audio(wavs[0], sr, fmt)
        return Response(
            content=audio,
            media_type=_MIME.get(fmt, "audio/wav"),
            headers={"X-Sample-Rate": str(sr)},
        )
    # Batch response: list of base64 strings
    clips = [
        base64.b64encode(manager.encode_audio(w, sr, fmt)).decode()
        for w in wavs
    ]
    return JSONResponse({"sample_rate": sr, "format": fmt, "audio": clips})


# ── Custom voice ──────────────────────────────────────────────────────────────

@router.post("/custom-voice", summary="Synthesise with a named speaker")
def custom_voice(req: CustomVoiceRequest):
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if manager.model_type != "CustomVoice":
        raise HTTPException(
            status_code=422,
            detail=f"Loaded model ({manager.model_type}) does not support custom-voice. "
                   "Load a *-CustomVoice model.",
        )
    try:
        language = (
            [l.value for l in req.language]
            if isinstance(req.language, list)
            else req.language.value
        )
        wavs, sr = manager.generate_custom_voice(
            text=req.text,
            language=language,
            speaker=req.speaker,
            instruct=req.instruct,
            max_new_tokens=req.max_new_tokens or config.MAX_NEW_TOKENS,
        )
    except Exception as exc:
        logger.exception("custom-voice error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _respond(wavs, sr, req.response_format.value)


# ── Voice design ──────────────────────────────────────────────────────────────

@router.post("/voice-design", summary="Design a voice from a text description")
def voice_design(req: VoiceDesignRequest):
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if manager.model_type != "VoiceDesign":
        raise HTTPException(
            status_code=422,
            detail=f"Loaded model ({manager.model_type}) does not support voice-design. "
                   "Load the *-VoiceDesign model.",
        )
    try:
        language = (
            [l.value for l in req.language]
            if isinstance(req.language, list)
            else req.language.value
        )
        instruct = (
            [i for i in req.instruct]
            if isinstance(req.instruct, list)
            else req.instruct
        )
        wavs, sr = manager.generate_voice_design(
            text=req.text,
            language=language,
            instruct=instruct,
            max_new_tokens=req.max_new_tokens or config.MAX_NEW_TOKENS,
        )
    except Exception as exc:
        logger.exception("voice-design error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _respond(wavs, sr, req.response_format.value)


# ── Voice clone ───────────────────────────────────────────────────────────────

@router.post("/voice-clone", summary="Clone a voice from a reference audio clip")
def voice_clone(req: VoiceCloneRequest):
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if manager.model_type != "Base":
        raise HTTPException(
            status_code=422,
            detail=f"Loaded model ({manager.model_type}) does not support voice-clone. "
                   "Load the *-Base model.",
        )
    try:
        language = (
            [l.value for l in req.language]
            if isinstance(req.language, list)
            else req.language.value
        )
        wavs, sr = manager.generate_voice_clone(
            text=req.text,
            language=language,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            max_new_tokens=req.max_new_tokens or config.MAX_NEW_TOKENS,
        )
    except Exception as exc:
        logger.exception("voice-clone error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _respond(wavs, sr, req.response_format.value)


# ── Metadata ──────────────────────────────────────────────────────────────────

class SpeakersResponse(BaseModel):
    speakers: List[str]
    native_languages: Dict[str, str]


@router.get("/speakers", summary="List available speakers")
def list_speakers():
    speakers = manager.get_supported_speakers()
    descriptions = {
        "Vivian": "Bright, slightly edgy young female voice (Chinese native)",
        "Serena": "Warm, gentle young female voice (Chinese native)",
        "Uncle_Fu": "Seasoned male voice with a low, mellow timbre (Chinese native)",
        "Dylan": "Youthful Beijing male voice, clear natural timbre (Chinese native)",
        "Eric": "Lively Chengdu male voice with slight husky brightness (Sichuan dialect native)",
        "Ryan": "Dynamic male voice with strong rhythmic drive (English native)",
        "Aiden": "Sunny American male voice with a clear midrange (English native)",
        "Ono_Anna": "Playful Japanese female voice with light, nimble timbre (Japanese native)",
        "Sohee": "Warm Korean female voice with rich emotion (Korean native)",
    }
    return {"speakers": speakers, "descriptions": descriptions}


@router.get("/languages", summary="List supported languages")
def list_languages():
    return {"languages": manager.get_supported_languages()}
