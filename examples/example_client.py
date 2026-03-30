#!/usr/bin/env python3
"""
example_client.py – Example API client for the Qwen3-TTS Voice-Design server.

Demonstrates the voice-design template system for audiobook production:
  1. OpenAI-compatible /v1/audio/speech with template IDs
  2. Extended /v1/tts/voice-design with templates, emotions, and pace
  3. Browsing available templates, emotions, and pace modifiers
  4. Multi-language storytelling (English, German, Portuguese)
  5. Character voice switching in a single scene
  6. Batch synthesis

Usage:
    python examples/example_client.py --base-url http://<server-ip>:8000
"""

import argparse
import base64
import json
import sys

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


def save_wav(data: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(data)
    print(f"  Saved → {path}")


def health_check(client: httpx.Client) -> None:
    print("\n── 0. Health check ────────────────────────────────────")
    r = client.get("/health/ready", timeout=10)
    print(" ", r.json())


# ── Template browsing ────────────────────────────────────────────────────────

def browse_templates(client: httpx.Client) -> None:
    """GET /v1/tts/voice-templates – explore available voice presets."""
    print("\n── 1. Browse voice-design templates ───────────────────")

    # List all categories
    r = client.get("/v1/tts/voice-templates", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"  Categories: {data['categories']}")
    print(f"  Total templates: {data['count']}")

    # Show narrator templates
    r = client.get("/v1/tts/voice-templates", params={"category": "narrator"}, timeout=10)
    r.raise_for_status()
    narrators = r.json()["templates"]
    print(f"\n  Narrator templates ({len(narrators)}):")
    for t in narrators:
        print(f"    • {t['id']:30s} – {t['description']}")

    # Show emotion modifiers
    r = client.get("/v1/tts/emotion-modifiers", timeout=10)
    r.raise_for_status()
    emotions = r.json()["modifiers"]
    print(f"\n  Emotion modifiers ({len(emotions)}):")
    for key in sorted(emotions):
        print(f"    • {key:15s} – {emotions[key][:60]}…")

    # Show pace modifiers
    r = client.get("/v1/tts/pace-modifiers", timeout=10)
    r.raise_for_status()
    paces = r.json()["modifiers"]
    print(f"\n  Pace modifiers ({len(paces)}):")
    for key, desc in paces.items():
        print(f"    • {key:15s} – {desc[:60]}…")


# ── OpenAI-compatible with template ──────────────────────────────────────────

def openai_with_template(client: httpx.Client) -> None:
    """POST /v1/audio/speech using a voice template ID."""
    print("\n── 2. OpenAI-compatible with template ─────────────────")
    r = client.post(
        "/v1/audio/speech",
        json={
            "input": (
                "Chapter One. The sun had barely risen over the misty hills "
                "when Eleanor stepped out of the cottage, her boots crunching "
                "on the frost-covered gravel path."
            ),
            "voice": "narrator-omniscient",
            "response_format": "wav",
            "language": "English",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_template_omniscient.wav")


# ── Voice design with emotions ───────────────────────────────────────────────

def voice_design_with_emotion(client: httpx.Client) -> None:
    """POST /v1/tts/voice-design – template + emotion + pace modifiers."""
    print("\n── 3. Voice design: villain with contempt ──────────────")
    r = client.post(
        "/v1/tts/voice-design",
        json={
            "text": (
                "You thought you could defeat me? How delightfully naive. "
                "I have been ten steps ahead of you from the very beginning."
            ),
            "language": "English",
            "template_id": "char-villain",
            "emotion": "contempt",
            "pace": "slow",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_villain_contempt.wav")

    print("\n── 3b. Voice design: hero with determination ───────────")
    r = client.post(
        "/v1/tts/voice-design",
        json={
            "text": (
                "We will not surrender. We will not retreat. "
                "Every soul in this city is counting on us, "
                "and we will not let them down."
            ),
            "language": "English",
            "template_id": "char-hero",
            "emotion": "determination",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_hero_determined.wav")

    print("\n── 3c. Voice design: bedtime narrator (peaceful) ───────")
    r = client.post(
        "/v1/tts/voice-design",
        json={
            "text": (
                "And so the little fox curled up beneath the great oak tree, "
                "listening to the whisper of the wind through the leaves, "
                "and slowly, ever so slowly, drifted off to sleep."
            ),
            "language": "English",
            "template_id": "narrator-bedtime",
            "emotion": "tenderness",
            "pace": "very-slow",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_bedtime_peaceful.wav")


# ── Multi-language storytelling ──────────────────────────────────────────────

def multilingual_narration(client: httpx.Client) -> None:
    """Voice design works across English, German, and Portuguese."""
    print("\n── 4. German narration (Hochdeutsch) ──────────────────")
    r = client.post(
        "/v1/tts/voice-design",
        json={
            "text": (
                "Erstes Kapitel. Es war einmal ein kleines Mädchen, "
                "das in einem dunklen Wald lebte. Jeden Morgen erwachte "
                "es mit dem Gesang der Vögel und dem Duft von frischem Moos."
            ),
            "language": "German",
            "template_id": "lang-de-narrator",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_german_narrator.wav")

    print("\n── 4b. Portuguese narration (Brazilian) ────────────────")
    r = client.post(
        "/v1/tts/voice-design",
        json={
            "text": (
                "Capítulo Um. O sol nascia lentamente sobre as montanhas, "
                "pintando o céu com tons de ouro e carmesim. Maria abriu "
                "a janela e respirou o ar fresco da manhã."
            ),
            "language": "Portuguese",
            "template_id": "lang-pt-br-narrator",
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_portuguese_narrator.wav")


# ── Character scene ──────────────────────────────────────────────────────────

def character_scene(client: httpx.Client) -> None:
    """Demonstrate switching voices for different characters in a scene."""
    print("\n── 5. Character scene: narrator + dialogue ─────────────")

    scene = [
        {
            "text": "The old wizard leaned forward, his eyes gleaming in the firelight.",
            "template_id": "narrator-epic-fantasy",
            "emotion": None,
            "filename": "/tmp/out_scene_narrator.wav",
        },
        {
            "text": "Listen carefully, child. What I am about to tell you will change everything you think you know.",
            "template_id": "char-mentor",
            "emotion": "reverence",
            "filename": "/tmp/out_scene_mentor.wav",
        },
        {
            "text": "But master, how can one person possibly make a difference against such darkness?",
            "template_id": "char-child-boy",
            "emotion": "fear",
            "filename": "/tmp/out_scene_child.wav",
        },
        {
            "text": "The wizard smiled, a deep, knowing smile that held centuries of wisdom.",
            "template_id": "narrator-epic-fantasy",
            "emotion": "tenderness",
            "filename": "/tmp/out_scene_narrator2.wav",
        },
    ]

    for i, line in enumerate(scene):
        r = client.post(
            "/v1/tts/voice-design",
            json={
                "text": line["text"],
                "language": "English",
                "template_id": line["template_id"],
                **({"emotion": line["emotion"]} if line["emotion"] else {}),
                "response_format": "wav",
            },
            timeout=120,
        )
        r.raise_for_status()
        save_wav(r.content, line["filename"])


# ── Custom instruct (no template) ────────────────────────────────────────────

def custom_instruct(client: httpx.Client) -> None:
    """Use a fully custom instruct without any template."""
    print("\n── 6. Fully custom voice instruct ──────────────────────")
    r = client.post(
        "/v1/tts/voice-design",
        json={
            "text": (
                "Breaking news from the capital. In an unprecedented move, "
                "the parliament has voted unanimously to approve the new "
                "environmental protection act."
            ),
            "language": "English",
            "instruct": (
                "A crisp, professional female news anchor voice, mid-30s, "
                "clear alto with perfect diction. Steady, authoritative "
                "delivery with the polished confidence of a veteran broadcaster."
            ),
            "response_format": "wav",
        },
        timeout=120,
    )
    r.raise_for_status()
    save_wav(r.content, "/tmp/out_custom_newsanchor.wav")


# ── Mood demonstrations ──────────────────────────────────────────────────────

def mood_showcase(client: httpx.Client) -> None:
    """Show how mood templates transform the same narrator voice."""
    print("\n── 7. Mood showcase: same text, different atmospheres ──")

    text = "She opened the door and stepped inside. The room was exactly as she remembered it."

    moods = [
        ("mood-suspense", "/tmp/out_mood_suspense.wav"),
        ("mood-romantic", "/tmp/out_mood_romantic.wav"),
        ("mood-eerie", "/tmp/out_mood_eerie.wav"),
        ("mood-peaceful", "/tmp/out_mood_peaceful.wav"),
    ]

    for template_id, filename in moods:
        r = client.post(
            "/v1/tts/voice-design",
            json={
                "text": text,
                "language": "English",
                "template_id": template_id,
                "response_format": "wav",
            },
            timeout=120,
        )
        r.raise_for_status()
        save_wav(r.content, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice-Design example client")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="Bearer token if TTS_API_KEY is set")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the template browsing (no audio generation).",
    )
    args = parser.parse_args()

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    with httpx.Client(base_url=args.base_url, headers=headers) as client:
        health_check(client)
        browse_templates(client)

        if not args.quick:
            openai_with_template(client)
            voice_design_with_emotion(client)
            multilingual_narration(client)
            character_scene(client)
            custom_instruct(client)
            mood_showcase(client)

    print("\nAll examples completed successfully.")


if __name__ == "__main__":
    main()
