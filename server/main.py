"""
Qwen3-TTS self-hosted API server.

Entry point:  python -m server.main
              (or via uvicorn: uvicorn server.main:app --host 0.0.0.0 --port 8000)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .model_manager import manager
from .routers import health_router, speech_router, tts_router

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("tts.main")


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Qwen3-TTS server …")
    manager.load(
        model_name=config.DEFAULT_MODEL if not config.MODEL_LOCAL_DIR else None,
        local_dir=config.MODEL_LOCAL_DIR,
    )
    yield
    logger.info("Shutting down …")
    manager.unload()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Qwen3-TTS API",
    description=(
        "Self-hosted Qwen3-TTS text-to-speech server with an "
        "OpenAI-compatible /v1/audio/speech endpoint and extended "
        "custom-voice, voice-design, and voice-clone capabilities."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all LAN origins (adjust for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(speech_router)
app.include_router(tts_router)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if config.WORKERS > 1:
        logger.warning(
            "TTS_WORKERS=%d is set but only 1 worker is safe with a single GPU. "
            "Multiple workers would each try to load the model, causing CUDA OOM errors. "
            "Forcing WORKERS=1.",
            config.WORKERS,
        )
    uvicorn.run(
        "server.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,
        log_level="info",
    )
