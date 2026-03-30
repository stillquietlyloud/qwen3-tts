from .health import router as health_router
from .speech import router as speech_router
from .tts import router as tts_router

__all__ = ["health_router", "speech_router", "tts_router"]
