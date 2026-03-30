"""
OpenAI-compatible TTS endpoint.

POST /v1/audio/speech
  body: { "model": "...", "input": "...", "voice": "narrator-omniscient", "response_format": "wav" }
  Returns: raw audio bytes with the appropriate Content-Type header.

This endpoint auto-routes to the correct generation method based on which
model variant is currently loaded.  For VoiceDesign models, the ``voice``
field can be a template ID, a free-form voice description, or left empty
to use the server default.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from .. import config
from ..model_manager import manager
from ..schemas import AudioFormat, Language, OpenAISpeechRequest
from ..voice_templates import build_instruct, get_template
from .auth import verify_api_key

logger = logging.getLogger("tts.speech")

router = APIRouter(tags=["speech"])

_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


def _resolve_voice_design_instruct(req: OpenAISpeechRequest) -> str:
    """Build the instruct string for a VoiceDesign request.

    Resolution order:
    1. If ``template_id`` is set, use the template instruct as the base.
    2. Else if ``voice`` matches a known template ID, use that template.
    3. Else if ``voice`` is a non-empty string, treat it as a raw instruct.
    4. Fall back to the server's DEFAULT_VOICE_INSTRUCT.

    ``emotion``, ``pace``, and ``instruct`` are always layered on top.
    """
    template_id = req.template_id
    base_instruct = None

    if not template_id and req.voice:
        # Check if voice is actually a template ID
        if get_template(req.voice):
            template_id = req.voice
        else:
            base_instruct = req.voice

    if not template_id and not base_instruct:
        base_instruct = config.DEFAULT_VOICE_INSTRUCT

    return build_instruct(
        template_id=template_id,
        base_instruct=base_instruct,
        emotion=req.emotion,
        pace=req.pace,
        extra=req.instruct,
    )


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
            instruct = _resolve_voice_design_instruct(req)
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    audio_bytes = manager.encode_audio(wavs[0], sr, fmt)
    return Response(
        content=audio_bytes,
        media_type=_MIME.get(fmt, "audio/wav"),
        headers={"X-Sample-Rate": str(sr)},
    )
