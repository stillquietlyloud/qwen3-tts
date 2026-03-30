"""
OpenAI-compatible TTS endpoint.

POST /v1/audio/speech
  body: { "model": "...", "input": "...", "voice": "Ryan", "response_format": "wav" }
  Returns: raw audio bytes with the appropriate Content-Type header.

This endpoint auto-routes to the correct generation method based on which
model variant is currently loaded.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from .. import config
from ..model_manager import manager
from ..schemas import AudioFormat, Language, OpenAISpeechRequest
from .auth import verify_api_key

logger = logging.getLogger("tts.speech")

router = APIRouter(tags=["speech"])

_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


@router.post(
    "/v1/audio/speech",
    dependencies=[Depends(verify_api_key)],
    summary="OpenAI-compatible text-to-speech",
)
def create_speech(req: OpenAISpeechRequest) -> Response:
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    language = (req.language or Language(config.DEFAULT_LANGUAGE)).value
    fmt = req.response_format.value
    max_new_tokens = config.MAX_NEW_TOKENS

    try:
        if manager.model_type == "VoiceDesign":
            # Use voice field as the instruct description
            instruct = req.instruct or req.voice or "A warm, clear English narrator."
            wavs, sr = manager.generate_voice_design(
                text=req.input,
                language=language,
                instruct=instruct,
                max_new_tokens=max_new_tokens,
            )
        elif manager.model_type == "Base":
            raise HTTPException(
                status_code=422,
                detail=(
                    "The Base model requires a reference audio for voice clone. "
                    "Use POST /v1/tts/voice-clone instead."
                ),
            )
        else:
            # CustomVoice (default)
            wavs, sr = manager.generate_custom_voice(
                text=req.input,
                language=language,
                speaker=req.voice or config.DEFAULT_SPEAKER,
                instruct=req.instruct,
                max_new_tokens=max_new_tokens,
            )
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    audio_bytes = manager.encode_audio(wavs[0], sr, fmt)
    return Response(
        content=audio_bytes,
        media_type=_MIME.get(fmt, "audio/wav"),
        headers={"X-Sample-Rate": str(sr)},
    )
