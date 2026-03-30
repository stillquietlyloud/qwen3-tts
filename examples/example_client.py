#!/usr/bin/env python3
"""
example_client.py – Example API client for the Qwen3-TTS server.

Demonstrates all three main endpoints:
  1. OpenAI-compatible /v1/audio/speech  (simplest, drop-in compatible)
  2. Extended /v1/tts/custom-voice       (explicit speaker + instruct)
  3. Extended /v1/tts/voice-design       (describe the voice in words)
  4. Extended /v1/tts/voice-clone        (clone from a 3-second clip)
  5. Batch synthesis

Usage:
    python examples/example_client.py --base-url http://<server-ip>:8000
"""

import argparse
import base64
import sys

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


def save_wav(data: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(data)
    print(f"  Saved → {path}")


def openai_compatible(client: httpx.Client) -> None:
    """POST /v1/audio/speech – drop-in OpenAI TTS replacement."""
    print("\n── 1. OpenAI-compatible endpoint ──────────────────────")
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "input": (
                "Chapter One. The sun had barely risen over the misty hills "
                "when Eleanor stepped out of the cottage, her boots crunching "
                "on the frost-covered gravel path."
            ),
            "voice": "Ryan",
            "response_format": "wav",
            "language": "English",
            "instruct": "Speak in a warm, measured narrator tone suitable for an audiobook.",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_openai.wav")


def custom_voice(client: httpx.Client) -> None:
    """POST /v1/tts/custom-voice – named speaker with optional style."""
    print("\n── 2. Custom voice (English – Ryan) ───────────────────")
    r = client.post(
        "/v1/tts/custom-voice",
        json={
            "text": (
                "In the beginning was the Word, and the Word was with God, "
                "and the Word was God."
            ),
            "language": "English",
            "speaker": "Ryan",
            "instruct": "Slow, reverential, like a Sunday morning sermon.",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_custom_voice_en.wav")

    print("\n── 2b. Custom voice (German – Ryan) ───────────────────")
    r = client.post(
        "/v1/tts/custom-voice",
        json={
            "text": (
                "Es war einmal ein kleines Mädchen, das in einem Wald lebte. "
                "Jeden Morgen erwachte sie mit dem Gesang der Vögel."
            ),
            "language": "German",
            "speaker": "Ryan",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_custom_voice_de.wav")

    print("\n── 2c. Custom voice (Portuguese – Aiden) ──────────────")
    r = client.post(
        "/v1/tts/custom-voice",
        json={
            "text": (
                "A vida é bela quando compartilhamos os nossos sonhos "
                "com aqueles que amamos."
            ),
            "language": "Portuguese",
            "speaker": "Aiden",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_custom_voice_pt.wav")


def voice_design(client: httpx.Client) -> None:
    """POST /v1/tts/voice-design – describe the voice you want.

    NOTE: This endpoint requires the VoiceDesign model variant.
    Set TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign before
    starting the server, otherwise this call returns HTTP 422.
    """
    print("\n── 3. Voice design ────────────────────────────────────")
    try:
        r = client.post(
            "/v1/tts/voice-design",
            json={
                "text": (
                    "Welcome, traveller. You have entered the Whispering Woods. "
                    "Few who venture here return unchanged."
                ),
                "language": "English",
                "instruct": (
                    "Deep, gravelly male voice, aged 60s, slow and deliberate, "
                    "like an ancient storyteller by a fire."
                ),
                "response_format": "wav",
            },
            timeout=120,
        )
        if r.status_code == 422:
            print(
                "  Skipped – VoiceDesign model not loaded. "
                "Set TTS_DEFAULT_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign to use this endpoint."
            )
            return
        r.raise_for_status()
        save_wav(r.content, "/tmp/out_voice_design.wav")
    except httpx.HTTPStatusError as exc:
        print(f"  HTTP error {exc.response.status_code}: {exc.response.text}")


def batch_synthesis(client: httpx.Client) -> None:
    """Batch: send multiple texts in one request."""
    print("\n── 4. Batch synthesis ─────────────────────────────────")
    r = client.post(
        "/v1/tts/custom-voice",
        json={
            "text": [
                "Chapter One. A new beginning.",
                "Chapter Two. The journey continues.",
                "Chapter Three. The final confrontation.",
            ],
            "language": ["English", "English", "English"],
            "speaker": ["Ryan", "Ryan", "Ryan"],
            "response_format": "wav",
        },
        timeout=300,
    )
    r.raise_for_status()
    payload = r.json()
    for i, clip_b64 in enumerate(payload["audio"]):
        data = base64.b64decode(clip_b64)
        save_wav(data, f"/tmp/out_batch_{i + 1}.wav")


def health_check(client: httpx.Client) -> None:
    print("\n── 0. Health check ────────────────────────────────────")
    r = client.get("/health/ready", timeout=10)
    print(" ", r.json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS API example client")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="Bearer token if TTS_API_KEY is set")
    args = parser.parse_args()

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    with httpx.Client(base_url=args.base_url, headers=headers) as client:
        health_check(client)
        openai_compatible(client)
        custom_voice(client)
        voice_design(client)
        batch_synthesis(client)

    print("\nAll examples completed successfully.")


if __name__ == "__main__":
    main()
