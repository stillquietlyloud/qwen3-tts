"""
Extended TTS endpoints that expose the full Qwen3-TTS feature set.

POST /v1/tts/custom-voice      – custom-speaker synthesis
POST /v1/tts/voice-design      – voice designed from a text description or template
POST /v1/tts/voice-clone       – 3-second voice clone from a reference clip
GET  /v1/tts/speakers          – list available speakers
GET  /v1/tts/languages         – list supported languages
GET  /v1/tts/voice-templates   – browse all voice-design templates
GET  /v1/tts/voice-templates/{id}  – get a single template
GET  /v1/tts/emotion-modifiers – list emotion modifiers
GET  /v1/tts/pace-modifiers    – list pace modifiers

All endpoints return raw audio bytes when the request contains a single
text string, or a JSON list of base64-encoded audio clips for batch
requests.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .. import config
from ..model_manager import manager
from ..schemas import (
    CustomVoiceRequest,
    VoiceCloneRequest,
    VoiceDesignRequest,
)
from ..voice_templates import (
    build_instruct,
    get_all_templates,
    get_categories,
    get_emotion_modifiers,
    get_pace_modifiers,
    get_template,
    get_templates_by_category,
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


def _resolve_voice_design_instruct(
    instruct: Optional[str],
    template_id: Optional[str],
    emotion: Optional[str],
    pace: Optional[str],
) -> str:
    """Build the final instruct string for a voice-design request."""
    if not template_id and not instruct:
        instruct = config.DEFAULT_VOICE_INSTRUCT
    return build_instruct(
        template_id=template_id,
        base_instruct=instruct if not template_id else None,
        emotion=emotion,
        pace=pace,
        extra=instruct if template_id else None,
    )


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

@router.post("/voice-design", summary="Design a voice from a text description or template")
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

        # Build the instruct — supports single or batch
        single_instruct = req.instruct if isinstance(req.instruct, str) or req.instruct is None else None
        if isinstance(req.instruct, list):
            # Batch: build per-item instructs (template/emotion/pace applied to each)
            instruct = [
                _resolve_voice_design_instruct(i, req.template_id, req.emotion, req.pace)
                for i in req.instruct
            ]
        else:
            instruct = _resolve_voice_design_instruct(
                single_instruct, req.template_id, req.emotion, req.pace,
            )

        wavs, sr = manager.generate_voice_design(
            text=req.text,
            language=language,
            instruct=instruct,
            max_new_tokens=req.max_new_tokens or config.MAX_NEW_TOKENS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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


# ── Voice templates ───────────────────────────────────────────────────────────

@router.get("/voice-templates", summary="List all voice-design templates")
def list_voice_templates(
    category: Optional[str] = Query(
        default=None,
        description="Filter by category: narrator, character, emotion, mood, style, language",
    ),
):
    """Return all available voice-design templates, optionally filtered by category."""
    if category:
        templates = get_templates_by_category(category)
    else:
        templates = get_all_templates()

    return {
        "categories": get_categories(),
        "count": len(templates),
        "templates": [t.to_dict() for t in templates],
    }


@router.get("/voice-templates/{template_id}", summary="Get a specific voice template")
def get_voice_template(template_id: str):
    """Return a single voice-design template by its ID."""
    tpl = get_template(template_id)
    if tpl is None:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_id}' not found. "
                   "Use GET /v1/tts/voice-templates to list all available templates.",
        )
    return tpl.to_dict()


@router.get("/emotion-modifiers", summary="List emotion modifier keywords")
def list_emotion_modifiers():
    """Return all emotion modifiers that can be applied to any voice template."""
    return {"modifiers": get_emotion_modifiers()}


@router.get("/pace-modifiers", summary="List pace modifier keywords")
def list_pace_modifiers():
    """Return all pace modifiers that can be applied to any voice template."""
    return {"modifiers": get_pace_modifiers()}


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
