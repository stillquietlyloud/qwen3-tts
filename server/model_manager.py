"""
Model lifecycle manager: loads the Qwen3-TTS model once at startup and
exposes thin wrappers used by the API routers.

Thread-safety: FastAPI runs handlers in a thread-pool (sync) or on the
event loop (async).  Because a single GPU cannot run two kernels at the
same time, inference is serialised via a threading.Lock so concurrent
requests are queued rather than producing CUDA OOM errors.

Stability: after every generation the CUDA cache is trimmed so that
long audiobook pipelines do not accumulate fragmented GPU memory.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import List, Optional, Tuple, Union

import numpy as np

from . import config

logger = logging.getLogger("tts.model_manager")


class ModelManager:
    """Singleton that owns the loaded Qwen3TTSModel."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._model_name: Optional[str] = None
        self._model_type: Optional[str] = None  # "CustomVoice" | "VoiceDesign" | "Base"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(
        self,
        model_name: Optional[str] = None,
        local_dir: Optional[str] = None,
    ) -> None:
        """Load (or reload) the TTS model onto the GPU."""
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore[import]

        model_id = model_name or local_dir or config.DEFAULT_MODEL
        dtype = torch.bfloat16 if config.DTYPE == "bfloat16" else torch.float16

        attn = "flash_attention_2" if config.USE_FLASH_ATTN else "eager"

        logger.info("Loading model %s …", model_id)
        with self._lock:
            self._model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=config.DEVICE,
                dtype=dtype,
                attn_implementation=attn,
            )
            self._model_name = model_id
            # Infer model type from the model name
            name_lower = model_id.lower()
            if "voicedesign" in name_lower:
                self._model_type = "VoiceDesign"
            elif "base" in name_lower:
                self._model_type = "Base"
            else:
                self._model_type = "CustomVoice"

        logger.info("Model ready  (type=%s  device=%s)", self._model_type, config.DEVICE)

    def unload(self) -> None:
        import torch

        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                torch.cuda.empty_cache()
                logger.info("Model unloaded.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    def get_supported_speakers(self) -> List[str]:
        if self._model is None or self._model_type != "CustomVoice":
            return []
        return list(self._model.get_supported_speakers())

    def get_supported_languages(self) -> List[str]:
        if self._model is None:
            return []
        return list(self._model.get_supported_languages())

    # ── CUDA housekeeping ─────────────────────────────────────────────────────

    @staticmethod
    def _trim_cuda_cache() -> None:
        """Free unused CUDA memory after each generation.

        For long audiobook pipelines this prevents fragmentation from slowly
        consuming all VRAM across hundreds of sequential generations.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Inference ─────────────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet.")

    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "English",
        speaker: Union[str, List[str]] = "Ryan",
        instruct: Optional[Union[str, List[str]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], int]:
        self._assert_loaded()
        kwargs: dict = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        with self._lock:
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or ([""] * len(text) if isinstance(text, list) else ""),
                **kwargs,
            )
            self._trim_cuda_cache()
        return wavs, sr

    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "English",
        instruct: Union[str, List[str]] = "",
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], int]:
        self._assert_loaded()
        kwargs: dict = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        with self._lock:
            wavs, sr = self._model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                **kwargs,
            )
            self._trim_cuda_cache()
        return wavs, sr

    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "English",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], int]:
        self._assert_loaded()
        kwargs: dict = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        x_vector_only = ref_text is None
        with self._lock:
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                **kwargs,
            )
            self._trim_cuda_cache()
        return wavs, sr

    # ── Audio encoding ────────────────────────────────────────────────────────

    @staticmethod
    def encode_audio(wav: np.ndarray, sr: int, fmt: str) -> bytes:
        """Convert a numpy waveform to bytes in the requested container format."""
        import soundfile as sf  # type: ignore[import]

        buf = io.BytesIO()
        if fmt == "mp3":
            # soundfile does not write MP3; use pydub if available, else WAV fallback
            try:
                from pydub import AudioSegment  # type: ignore[import]

                tmp = io.BytesIO()
                sf.write(tmp, wav, sr, format="WAV")
                tmp.seek(0)
                seg = AudioSegment.from_wav(tmp)
                seg.export(buf, format="mp3", bitrate="320k")
            except ImportError:
                logger.warning("pydub not installed; falling back to WAV for MP3 request.")
                sf.write(buf, wav, sr, format="WAV")
        elif fmt == "ogg":
            sf.write(buf, wav, sr, format="OGG", subtype="VORBIS")
        elif fmt == "flac":
            sf.write(buf, wav, sr, format="FLAC")
        else:
            sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        return buf.read()


# ── Module-level singleton ────────────────────────────────────────────────────
manager = ModelManager()
