"""
Voice design templates for immersive audiobook storytelling.

This module provides a comprehensive library of voice-design instruction
templates that clients can use directly or combine to create unique voices
for every character, mood, and situation in a long-form audiobook pipeline.

Each template is a plain-English description understood by the
Qwen3-TTS-VoiceDesign model's ``instruct`` parameter.

Categories
----------
- **Narrators** – main storytelling voices (omniscient, first-person, …)
- **Character archetypes** – hero, villain, mentor, child, …
- **Emotions** – joy, sadness, anger, fear, surprise, …
- **Moods / atmospheres** – suspense, romance, comedy, horror, …
- **Speech styles** – whisper, shout, inner-thought, …
- **Languages** – English, German, Portuguese accent/locale variants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceTemplate:
    """A single reusable voice-design preset."""

    id: str
    name: str
    category: str
    instruct: str
    description: str
    language_hint: str = "English"
    tags: tuple = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "instruct": self.instruct,
            "description": self.description,
            "language_hint": self.language_hint,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, VoiceTemplate] = {}


def _r(t: VoiceTemplate) -> VoiceTemplate:
    """Register a template and return it."""
    _TEMPLATES[t.id] = t
    return t


# ── Narrators ─────────────────────────────────────────────────────────────

_r(VoiceTemplate(
    id="narrator-omniscient",
    name="Omniscient Narrator",
    category="narrator",
    instruct=(
        "A mature male voice, mid-40s, deep and resonant baritone with a warm, "
        "rich timbre. Measured, unhurried pace with subtle dramatic pauses. "
        "Authoritative yet inviting, like a master storyteller who has seen "
        "everything and speaks with quiet confidence."
    ),
    description="Classic third-person omniscient narrator for literary fiction.",
    tags=("male", "mature", "authoritative", "warm", "baritone"),
))

_r(VoiceTemplate(
    id="narrator-first-person",
    name="First-Person Narrator",
    category="narrator",
    instruct=(
        "A conversational male voice, early 30s, natural midrange tenor. "
        "Intimate and reflective, as if confiding in a close friend. "
        "Slightly informal cadence with personal warmth and occasional "
        "hesitations that feel genuine and unscripted."
    ),
    description="Intimate first-person narrator, ideal for memoirs and confessional fiction.",
    tags=("male", "conversational", "intimate", "tenor"),
))

_r(VoiceTemplate(
    id="narrator-female-literary",
    name="Female Literary Narrator",
    category="narrator",
    instruct=(
        "An elegant female voice, late 30s, clear mezzo-soprano with a smooth, "
        "polished timbre. Articulate and poised, with a gentle British-influenced "
        "cadence. Thoughtful pacing, emphasising key phrases with subtle "
        "warmth and intelligence."
    ),
    description="Polished female narrator for literary and historical fiction.",
    tags=("female", "elegant", "mezzo-soprano", "british", "literary"),
))

_r(VoiceTemplate(
    id="narrator-documentary",
    name="Documentary Narrator",
    category="narrator",
    instruct=(
        "A deep, commanding male voice, 50s, with a gravelly edge and "
        "authoritative delivery. Steady, deliberate pace with clear "
        "enunciation. Neutral accent, professional and serious, like a "
        "veteran news anchor narrating a world-changing event."
    ),
    description="Authoritative documentary-style narration for non-fiction.",
    tags=("male", "deep", "commanding", "professional", "documentary"),
))

_r(VoiceTemplate(
    id="narrator-bedtime",
    name="Bedtime Story Narrator",
    category="narrator",
    instruct=(
        "A soft, soothing female voice, mid-30s, gentle and melodic alto. "
        "Very slow, lulling pace with a dreamy quality. Warm and comforting, "
        "like a loving parent reading to a child at night. Quiet volume, "
        "tender inflections."
    ),
    description="Gentle voice for children's bedtime stories.",
    tags=("female", "soft", "soothing", "slow", "bedtime"),
))

_r(VoiceTemplate(
    id="narrator-thriller",
    name="Thriller Narrator",
    category="narrator",
    instruct=(
        "A tense, controlled male voice, late 30s, low baritone with a "
        "slight rasp. Quick, clipped delivery with sharp pauses that build "
        "suspense. Understated intensity, as if every word carries hidden "
        "danger. Restrained energy barely containing urgency."
    ),
    description="Suspenseful narrator for thrillers and crime fiction.",
    tags=("male", "tense", "raspy", "thriller", "suspense"),
))

_r(VoiceTemplate(
    id="narrator-epic-fantasy",
    name="Epic Fantasy Narrator",
    category="narrator",
    instruct=(
        "A grand, sonorous male voice, 50s, rich and powerful bass-baritone. "
        "Majestic, sweeping delivery with dramatic flourishes. Speaks as if "
        "recounting legends passed down through generations. Slow, deliberate "
        "pacing with reverent weight on every sentence."
    ),
    description="Grand narrator for epic fantasy and mythology.",
    tags=("male", "grand", "bass-baritone", "epic", "majestic"),
))

_r(VoiceTemplate(
    id="narrator-humorous",
    name="Humorous Narrator",
    category="narrator",
    instruct=(
        "A lively, witty male voice, mid-30s, bright tenor with playful "
        "inflections. Quick-witted delivery with comedic timing, occasional "
        "sarcastic undertones, and a mischievous sparkle. Light and energetic, "
        "as if sharing an amusing anecdote at a dinner party."
    ),
    description="Witty, light-hearted narrator for comedy and satire.",
    tags=("male", "witty", "playful", "comedy", "tenor"),
))

_r(VoiceTemplate(
    id="narrator-noir",
    name="Noir Narrator",
    category="narrator",
    instruct=(
        "A world-weary male voice, 40s, low and smoky baritone with a "
        "cynical edge. Slow, drawling delivery dripping with disillusionment. "
        "Speaks like a hardboiled detective recounting a case from a dimly "
        "lit bar. Laconic, dry, with bitter undertones."
    ),
    description="Hardboiled narrator for noir and detective fiction.",
    tags=("male", "smoky", "cynical", "noir", "world-weary"),
))

_r(VoiceTemplate(
    id="narrator-romance",
    name="Romance Narrator",
    category="narrator",
    instruct=(
        "A warm, sensual female voice, early 30s, velvety mezzo-soprano "
        "with a breathy, intimate quality. Unhurried, flowing delivery "
        "that lingers on emotional moments. Tender and passionate, with "
        "gentle rises and falls that mirror the heart's rhythm."
    ),
    description="Sensual, intimate narrator for romance novels.",
    tags=("female", "warm", "sensual", "breathy", "romance"),
))

_r(VoiceTemplate(
    id="narrator-horror",
    name="Horror Narrator",
    category="narrator",
    instruct=(
        "A chilling, quiet male voice, 40s, hollow midrange with an eerie, "
        "unsettling quality. Deliberate, creeping pace that slows at moments "
        "of dread. Whisper-adjacent delivery that makes every word feel like "
        "a warning. Cold, detached, with occasional sinister inflections."
    ),
    description="Unsettling narrator for horror and dark fiction.",
    tags=("male", "chilling", "eerie", "horror", "whisper"),
))

_r(VoiceTemplate(
    id="narrator-young-adult",
    name="Young Adult Narrator",
    category="narrator",
    instruct=(
        "A fresh, energetic female voice, late teens to early 20s, bright "
        "and clear soprano. Natural, relatable delivery with contemporary "
        "cadence. Emotionally expressive with authentic enthusiasm, "
        "vulnerability, and the occasional dramatic flair of youth."
    ),
    description="Youthful narrator for young adult fiction.",
    tags=("female", "young", "energetic", "bright", "soprano"),
))

# ── Character archetypes ──────────────────────────────────────────────────

_r(VoiceTemplate(
    id="char-hero",
    name="Hero",
    category="character",
    instruct=(
        "A strong, confident male voice, early 30s, clear and resonant "
        "midrange. Firm and determined delivery, radiating courage and "
        "nobility. Measured pace with conviction in every word. "
        "Natural leadership quality without arrogance."
    ),
    description="Classic hero archetype – brave, noble, determined.",
    tags=("male", "strong", "confident", "hero"),
))

_r(VoiceTemplate(
    id="char-heroine",
    name="Heroine",
    category="character",
    instruct=(
        "A bold, clear female voice, late 20s, strong mezzo-soprano with a "
        "steely edge. Determined and fierce, yet capable of great warmth. "
        "Direct delivery with unwavering confidence and emotional depth."
    ),
    description="Female hero archetype – fierce, determined, empathetic.",
    tags=("female", "bold", "fierce", "heroine"),
))

_r(VoiceTemplate(
    id="char-villain",
    name="Villain",
    category="character",
    instruct=(
        "A menacing, silky male voice, 40s, low and smooth with a predatory "
        "calm. Deliberate, controlled delivery dripping with veiled threat. "
        "Each word chosen with calculated precision. Charming on the surface "
        "but cold underneath."
    ),
    description="Antagonist – calculating, menacing, charismatic evil.",
    tags=("male", "menacing", "smooth", "villain", "calculating"),
))

_r(VoiceTemplate(
    id="char-villainess",
    name="Villainess",
    category="character",
    instruct=(
        "A sharp, commanding female voice, 30s, cold and precise with a "
        "venomous sweetness. Controlled delivery that can shift from honey "
        "to ice in a heartbeat. Elegant diction with an undercurrent of "
        "danger and contempt."
    ),
    description="Female antagonist – cunning, elegant, ruthless.",
    tags=("female", "sharp", "cold", "villainess", "cunning"),
))

_r(VoiceTemplate(
    id="char-mentor",
    name="Wise Mentor",
    category="character",
    instruct=(
        "An elderly male voice, 70s, deep and weathered with gentle "
        "warmth. Slow, thoughtful delivery full of patience and wisdom. "
        "Speaks as if every word has been considered for decades. "
        "Kindly but with gravity, like a sage imparting ancient knowledge."
    ),
    description="Wise old mentor – patient, knowledgeable, fatherly.",
    tags=("male", "elderly", "wise", "deep", "mentor"),
))

_r(VoiceTemplate(
    id="char-child-boy",
    name="Young Boy",
    category="character",
    instruct=(
        "A bright, innocent young male voice, about 8 years old, high-pitched "
        "and clear. Energetic and curious with a natural, unpolished delivery. "
        "Quick pace with breathless excitement. Pure and guileless."
    ),
    description="Child character – innocent, curious, energetic.",
    tags=("male", "child", "high-pitched", "innocent", "bright"),
))

_r(VoiceTemplate(
    id="char-child-girl",
    name="Young Girl",
    category="character",
    instruct=(
        "A sweet, high-pitched young female voice, about 9 years old, "
        "light and musical. Cheerful and animated with exaggerated "
        "expressions. Speaks with wonder and imagination, full of "
        "questions and excitement."
    ),
    description="Girl character – sweet, imaginative, lively.",
    tags=("female", "child", "sweet", "high-pitched", "lively"),
))

_r(VoiceTemplate(
    id="char-trickster",
    name="Trickster",
    category="character",
    instruct=(
        "A sly, quick male voice, 30s, light and nimble tenor with a "
        "mischievous lilt. Rapid, unpredictable delivery that darts between "
        "playful and devious. Always sounds like it knows something you "
        "do not. Cunning charm with a wink in every word."
    ),
    description="Trickster archetype – cunning, playful, unpredictable.",
    tags=("male", "sly", "quick", "trickster", "mischievous"),
))

_r(VoiceTemplate(
    id="char-warrior",
    name="Battle-Hardened Warrior",
    category="character",
    instruct=(
        "A rough, gravelly male voice, late 40s, low and battle-scarred. "
        "Blunt, terse delivery stripped of all pretence. Speaks in short, "
        "punchy sentences. Exhausted but unbreakable, with the quiet "
        "authority of someone who has survived countless fights."
    ),
    description="Veteran warrior – tough, blunt, weathered.",
    tags=("male", "rough", "gravelly", "warrior", "tough"),
))

_r(VoiceTemplate(
    id="char-royal",
    name="Royal / Noble",
    category="character",
    instruct=(
        "A refined, aristocratic male voice, 40s, cultured baritone with "
        "impeccable diction. Measured, formal delivery radiating authority "
        "and breeding. Every syllable polished, every pause deliberate. "
        "Speaks as if addressing a court."
    ),
    description="Royalty or noble character – refined, formal, commanding.",
    tags=("male", "refined", "aristocratic", "formal", "royal"),
))

_r(VoiceTemplate(
    id="char-mystic",
    name="Mystic / Oracle",
    category="character",
    instruct=(
        "An ethereal, androgynous voice, ageless, with a haunting, "
        "otherworldly quality. Slow, hypnotic delivery that seems to echo "
        "from another realm. Breathy and mysterious, with elongated vowels "
        "and a dreamlike cadence."
    ),
    description="Mystical being – ethereal, prophetic, otherworldly.",
    tags=("androgynous", "ethereal", "mystic", "haunting", "slow"),
))

_r(VoiceTemplate(
    id="char-companion",
    name="Loyal Companion",
    category="character",
    instruct=(
        "A warm, dependable male voice, mid-30s, friendly midrange tenor. "
        "Earnest and supportive delivery with a hint of gentle humour. "
        "Speaks with loyalty and care, always reassuring. Natural and "
        "unaffected, like a trusted best friend."
    ),
    description="Sidekick / companion – loyal, warm, reliable.",
    tags=("male", "warm", "friendly", "companion", "supportive"),
))

_r(VoiceTemplate(
    id="char-femme-fatale",
    name="Femme Fatale",
    category="character",
    instruct=(
        "A sultry, low female voice, 30s, smoky contralto with a seductive "
        "purr. Slow, deliberate delivery that drips with dangerous allure. "
        "Every word is a trap wrapped in velvet. Confident, mysterious, "
        "and utterly in control."
    ),
    description="Seductive, dangerous female character.",
    tags=("female", "sultry", "smoky", "seductive", "contralto"),
))

_r(VoiceTemplate(
    id="char-comic-relief",
    name="Comic Relief",
    category="character",
    instruct=(
        "A bumbling, high-energy male voice, 30s, nasal and slightly "
        "squeaky tenor. Fast, excitable delivery prone to stumbling over "
        "words. Exaggerated reactions, nervous energy, and endearing "
        "clumsiness. Always slightly out of breath."
    ),
    description="Comic relief character – bumbling, excitable, lovable.",
    tags=("male", "high-energy", "nasal", "comedy", "bumbling"),
))

_r(VoiceTemplate(
    id="char-elder-woman",
    name="Elder Woman",
    category="character",
    instruct=(
        "An aged female voice, late 70s, thin and reedy but full of "
        "quiet strength. Slow, deliberate delivery with a grandmother's "
        "gentle patience. Occasionally trembles but never wavers in "
        "conviction. Warm, wise, and deeply compassionate."
    ),
    description="Elderly female – wise, gentle, resilient.",
    tags=("female", "elderly", "thin", "gentle", "wise"),
))

# ── Emotions ──────────────────────────────────────────────────────────────

_r(VoiceTemplate(
    id="emotion-joy",
    name="Joyful",
    category="emotion",
    instruct=(
        "Speak with bright, infectious joy. Voice lifts upward with "
        "enthusiasm, pace quickens with excitement. Warm, beaming quality "
        "as if smiling broadly. Light and buoyant delivery, radiating "
        "pure happiness and delight."
    ),
    description="Pure joy and delight – bright, warm, uplifting.",
    tags=("joyful", "bright", "warm", "fast"),
))

_r(VoiceTemplate(
    id="emotion-sadness",
    name="Sorrowful",
    category="emotion",
    instruct=(
        "Speak with deep, quiet sadness. Voice drops lower, pace slows "
        "considerably. Occasional trembling quality, words trailing off "
        "with resignation. Heavy with grief, each breath carries the "
        "weight of loss. Restrained, not melodramatic."
    ),
    description="Deep sadness and grief – slow, heavy, trembling.",
    tags=("sad", "slow", "heavy", "grief"),
))

_r(VoiceTemplate(
    id="emotion-anger",
    name="Angry",
    category="emotion",
    instruct=(
        "Speak with controlled, simmering anger. Voice tightens, words "
        "become clipped and forceful. Rising intensity with sharp "
        "consonants. Barely contained fury that threatens to break through "
        "at any moment. Hard, biting delivery."
    ),
    description="Controlled anger – tight, forceful, simmering.",
    tags=("angry", "forceful", "sharp", "intense"),
))

_r(VoiceTemplate(
    id="emotion-fear",
    name="Frightened",
    category="emotion",
    instruct=(
        "Speak with palpable fear and anxiety. Voice shakes and rises "
        "in pitch. Quickened, breathless pace with nervous pauses. "
        "Words tumble out unevenly, as if the speaker's heart is "
        "pounding. Tight, constricted throat quality."
    ),
    description="Fear and anxiety – shaky, breathless, high-pitched.",
    tags=("fear", "shaky", "breathless", "nervous"),
))

_r(VoiceTemplate(
    id="emotion-surprise",
    name="Surprised",
    category="emotion",
    instruct=(
        "Speak with genuine surprise and wonder. Voice rises sharply "
        "in pitch, eyes widening. Short, breathless exclamations followed "
        "by rapid processing. A mixture of disbelief and amazement, "
        "caught completely off guard."
    ),
    description="Shock and amazement – sharp pitch rise, breathless.",
    tags=("surprised", "breathless", "sharp", "amazed"),
))

_r(VoiceTemplate(
    id="emotion-tenderness",
    name="Tender",
    category="emotion",
    instruct=(
        "Speak with gentle tenderness and affection. Soft, quiet voice "
        "barely above a whisper. Slow, careful delivery as if each word "
        "is a caress. Warm and protective, full of unspoken love. "
        "Intimate and deeply caring."
    ),
    description="Gentle tenderness – soft, intimate, loving.",
    tags=("tender", "soft", "intimate", "loving", "slow"),
))

_r(VoiceTemplate(
    id="emotion-determination",
    name="Determined",
    category="emotion",
    instruct=(
        "Speak with unwavering determination and resolve. Voice is firm, "
        "steady, and grounded. Measured pace with iron conviction behind "
        "every word. No hesitation, no doubt. Building intensity that "
        "inspires and commands respect."
    ),
    description="Steel resolve – firm, steady, inspiring.",
    tags=("determined", "firm", "steady", "resolute"),
))

_r(VoiceTemplate(
    id="emotion-despair",
    name="Despairing",
    category="emotion",
    instruct=(
        "Speak with hollow, broken despair. Voice is empty and flat, "
        "drained of all energy. Slow, monotone delivery as if hope has "
        "been completely extinguished. Occasionally cracks with raw "
        "emotion before retreating into numbness."
    ),
    description="Total despair – hollow, broken, empty.",
    tags=("despair", "hollow", "flat", "broken"),
))

_r(VoiceTemplate(
    id="emotion-awe",
    name="Awestruck",
    category="emotion",
    instruct=(
        "Speak with breathless awe and wonder. Voice drops to a reverent "
        "near-whisper, pace slows dramatically. Every word chosen with "
        "care, as if describing something sacred. Wide-eyed and humbled, "
        "barely able to articulate the magnificence witnessed."
    ),
    description="Reverent awe – breathless, slow, hushed.",
    tags=("awe", "reverent", "slow", "whisper", "wonder"),
))

_r(VoiceTemplate(
    id="emotion-contempt",
    name="Contemptuous",
    category="emotion",
    instruct=(
        "Speak with cutting contempt and disdain. Voice drips with "
        "superiority and dismissal. Slow, drawling delivery that "
        "emphasises each syllable of scorn. Cold and mocking, with "
        "a sneer audible in every inflection."
    ),
    description="Disdain and scorn – cold, mocking, superior.",
    tags=("contempt", "cold", "mocking", "disdain"),
))

# ── Moods / atmospheres ──────────────────────────────────────────────────

_r(VoiceTemplate(
    id="mood-suspense",
    name="Suspenseful",
    category="mood",
    instruct=(
        "Deliver with building suspense and tension. Low, hushed voice "
        "that tightens with each sentence. Deliberately slow pace with "
        "agonising pauses. Words hanging in the air, each one raising "
        "the stakes. Controlled but electric with hidden danger."
    ),
    description="Building tension and dread.",
    tags=("suspense", "tense", "slow", "hushed"),
))

_r(VoiceTemplate(
    id="mood-romantic",
    name="Romantic",
    category="mood",
    instruct=(
        "Deliver with warm, romantic atmosphere. Soft, flowing voice "
        "with a dreamy quality. Unhurried pace that savours each moment. "
        "Gentle rises and falls that mirror desire and longing. "
        "Intimate and heartfelt, like candlelight made audible."
    ),
    description="Warm, intimate romantic atmosphere.",
    tags=("romantic", "soft", "warm", "dreamy", "intimate"),
))

_r(VoiceTemplate(
    id="mood-action",
    name="Action / Combat",
    category="mood",
    instruct=(
        "Deliver with explosive, adrenaline-fueled energy. Rapid-fire "
        "pace that races with the action. Sharp, punchy delivery with "
        "visceral impact. Breathless urgency, as if narrating events "
        "unfolding at breakneck speed. Raw, kinetic power."
    ),
    description="High-energy action sequences.",
    tags=("action", "fast", "energetic", "punchy", "urgent"),
))

_r(VoiceTemplate(
    id="mood-melancholy",
    name="Melancholy",
    category="mood",
    instruct=(
        "Deliver with bittersweet melancholy. Voice carries a gentle "
        "ache, beautiful in its sadness. Slow, reflective pace that "
        "lingers on memories. Not despairing but wistful, like autumn "
        "light through rain-streaked glass."
    ),
    description="Wistful, bittersweet sadness.",
    tags=("melancholy", "wistful", "slow", "reflective"),
))

_r(VoiceTemplate(
    id="mood-triumph",
    name="Triumphant",
    category="mood",
    instruct=(
        "Deliver with swelling triumph and victory. Voice rises with "
        "power and pride, crescendoing with exultation. Pace builds "
        "from measured to soaring. Grand, celebratory energy that "
        "captures the moment of hard-won glory."
    ),
    description="Victory and celebration.",
    tags=("triumph", "powerful", "rising", "celebratory"),
))

_r(VoiceTemplate(
    id="mood-eerie",
    name="Eerie / Uncanny",
    category="mood",
    instruct=(
        "Deliver with unsettling, eerie atmosphere. Voice takes on a "
        "hollow, distant quality as if echoing through empty halls. "
        "Unnaturally even pace that refuses to rush despite growing "
        "dread. Something is deeply wrong and the voice knows it."
    ),
    description="Unsettling, otherworldly dread.",
    tags=("eerie", "hollow", "unsettling", "slow"),
))

_r(VoiceTemplate(
    id="mood-peaceful",
    name="Peaceful",
    category="mood",
    instruct=(
        "Deliver with serene, tranquil peace. Gentle, unhurried voice "
        "that flows like a calm stream. Slow, meditative pace with "
        "easy breathing between phrases. Warm and grounded, radiating "
        "contentment and quiet beauty."
    ),
    description="Calm, serene tranquillity.",
    tags=("peaceful", "gentle", "slow", "serene", "calm"),
))

_r(VoiceTemplate(
    id="mood-epic-battle",
    name="Epic Battle",
    category="mood",
    instruct=(
        "Deliver with the grandeur of an epic battle. Voice swells "
        "with gravity and heroic intensity. Powerful, sweeping delivery "
        "that alternates between thunderous charges and breathless pauses. "
        "The fate of worlds hangs in every syllable."
    ),
    description="Climactic battle with high stakes.",
    tags=("epic", "battle", "powerful", "grand", "intense"),
))

# ── Speech styles ─────────────────────────────────────────────────────────

_r(VoiceTemplate(
    id="style-whisper",
    name="Whisper",
    category="style",
    instruct=(
        "Speak in a hushed, urgent whisper. Voice barely above silence, "
        "breathy and intimate. Every consonant softened, every vowel "
        "compressed. Feels secret and dangerous, as if walls have ears."
    ),
    description="Hushed whisper – secretive, intimate, urgent.",
    tags=("whisper", "hushed", "breathy", "quiet"),
))

_r(VoiceTemplate(
    id="style-shout",
    name="Shout / Command",
    category="style",
    instruct=(
        "Speak with a commanding shout. Voice at full power, resonant "
        "and forceful. Sharp, percussive delivery that demands immediate "
        "attention. Authoritative and unyielding, cutting through chaos."
    ),
    description="Powerful shout or battle command.",
    tags=("shout", "loud", "commanding", "forceful"),
))

_r(VoiceTemplate(
    id="style-inner-thought",
    name="Inner Thought",
    category="style",
    instruct=(
        "Speak as an internal monologue. Quiet, contemplative voice "
        "that feels private and unspoken. Slower pace with natural "
        "pauses for reflection. Intimate and unguarded, thoughts "
        "forming in real time without performance."
    ),
    description="Internal monologue – private, reflective, unguarded.",
    tags=("inner-thought", "quiet", "reflective", "intimate"),
))

_r(VoiceTemplate(
    id="style-letter-reading",
    name="Reading a Letter",
    category="style",
    instruct=(
        "Speak as if reading a personal letter aloud. Careful, measured "
        "delivery that honours each handwritten word. Pauses between "
        "sentences as if turning a page. Emotional weight varies with "
        "the content, deeply personal and unrushed."
    ),
    description="Reading correspondence aloud – careful, personal.",
    tags=("letter", "careful", "measured", "personal"),
))

_r(VoiceTemplate(
    id="style-prayer",
    name="Prayer / Incantation",
    category="style",
    instruct=(
        "Speak with the reverent cadence of a prayer or ritual chant. "
        "Low, solemn voice with rhythmic, almost musical phrasing. "
        "Deliberate repetition and measured pauses. Sacred and ancient, "
        "as if invoking forces beyond mortal understanding."
    ),
    description="Ritual chant or prayer – solemn, rhythmic, sacred.",
    tags=("prayer", "solemn", "rhythmic", "sacred", "low"),
))

_r(VoiceTemplate(
    id="style-proclamation",
    name="Royal Proclamation",
    category="style",
    instruct=(
        "Speak with the authority of a royal decree. Booming, resonant "
        "voice projecting to a vast hall. Formal, archaic cadence with "
        "deliberate weight on every word. Grand and unquestionable, "
        "the voice of sovereign power."
    ),
    description="Formal royal decree – booming, authoritative, grand.",
    tags=("proclamation", "formal", "booming", "authoritative"),
))

_r(VoiceTemplate(
    id="style-sarcastic",
    name="Sarcastic",
    category="style",
    instruct=(
        "Speak with dry, biting sarcasm. Flat delivery with exaggerated "
        "emphasis on key words that betrays the speaker's true feelings. "
        "Deliberately unimpressed tone with a subtle eye-roll quality. "
        "Witty and cutting."
    ),
    description="Dry sarcasm – flat, cutting, deliberately unimpressed.",
    tags=("sarcastic", "dry", "flat", "witty"),
))

_r(VoiceTemplate(
    id="style-dialogue-excited",
    name="Excited Dialogue",
    category="style",
    instruct=(
        "Speak with breathless excitement in conversation. Rapid pace, "
        "words tumbling over each other. Rising pitch with enthusiastic "
        "energy. Genuine, uncontainable eagerness that makes the "
        "listener lean in."
    ),
    description="Excited character dialogue – fast, eager, rising pitch.",
    tags=("excited", "fast", "dialogue", "eager"),
))

# ── Language-specific variants ────────────────────────────────────────────

_r(VoiceTemplate(
    id="lang-en-british-narrator",
    name="British English Narrator",
    category="language",
    instruct=(
        "A refined British male voice, mid-40s, received pronunciation with "
        "a warm, cultured baritone. Measured, elegant delivery with crisp "
        "consonants and rounded vowels. Evokes tradition, intelligence, "
        "and understated authority. Perfect BBC cadence."
    ),
    description="British RP narrator for English audiobooks.",
    language_hint="English",
    tags=("male", "british", "rp", "refined", "english"),
))

_r(VoiceTemplate(
    id="lang-en-american-narrator",
    name="American English Narrator",
    category="language",
    instruct=(
        "A friendly American male voice, mid-30s, clear General American "
        "accent with a natural, approachable tenor. Conversational and "
        "engaging delivery, neither too formal nor too casual. "
        "Warm and relatable, like public radio at its best."
    ),
    description="General American narrator for English audiobooks.",
    language_hint="English",
    tags=("male", "american", "friendly", "natural", "english"),
))

_r(VoiceTemplate(
    id="lang-de-narrator",
    name="German Narrator (Hochdeutsch)",
    category="language",
    instruct=(
        "A clear, articulate male voice, mid-40s, speaking standard "
        "Hochdeutsch with a rich baritone timbre. Measured, precise "
        "delivery with clean enunciation of compound words. Warm yet "
        "authoritative, suitable for literary German narration. "
        "Natural rhythm respecting German sentence structure."
    ),
    description="Standard German (Hochdeutsch) narrator.",
    language_hint="German",
    tags=("male", "german", "hochdeutsch", "clear", "baritone"),
))

_r(VoiceTemplate(
    id="lang-de-female-narrator",
    name="German Female Narrator",
    category="language",
    instruct=(
        "A warm, expressive female voice, mid-30s, clear Hochdeutsch "
        "mezzo-soprano. Elegant, flowing delivery that handles long "
        "German compound sentences with grace. Intelligent and engaging, "
        "with natural warmth and precise diction."
    ),
    description="Female German narrator for Hochdeutsch audiobooks.",
    language_hint="German",
    tags=("female", "german", "hochdeutsch", "warm", "mezzo-soprano"),
))

_r(VoiceTemplate(
    id="lang-pt-br-narrator",
    name="Brazilian Portuguese Narrator",
    category="language",
    instruct=(
        "A warm, melodic male voice, mid-30s, speaking clear Brazilian "
        "Portuguese with a rich, musical quality. Natural carioca-influenced "
        "cadence with smooth, flowing delivery. Engaging and expressive, "
        "with the characteristic warmth and rhythm of Brazilian speech."
    ),
    description="Brazilian Portuguese narrator.",
    language_hint="Portuguese",
    tags=("male", "portuguese", "brazilian", "warm", "melodic"),
))

_r(VoiceTemplate(
    id="lang-pt-eu-narrator",
    name="European Portuguese Narrator",
    category="language",
    instruct=(
        "A clear, refined male voice, mid-40s, speaking standard European "
        "Portuguese with precise Lisbon-influenced diction. More clipped "
        "and consonant-heavy than Brazilian, with a dignified, literary "
        "quality. Measured pace, articulate and cultured."
    ),
    description="European Portuguese narrator.",
    language_hint="Portuguese",
    tags=("male", "portuguese", "european", "refined", "precise"),
))

_r(VoiceTemplate(
    id="lang-pt-female-narrator",
    name="Portuguese Female Narrator",
    category="language",
    instruct=(
        "A vibrant, expressive female voice, early 30s, speaking Portuguese "
        "with a warm, melodic mezzo-soprano. Natural, flowing delivery with "
        "emotional expressiveness. Rich and engaging, capturing the musicality "
        "of the Portuguese language."
    ),
    description="Female Portuguese narrator.",
    language_hint="Portuguese",
    tags=("female", "portuguese", "vibrant", "melodic", "mezzo-soprano"),
))

# ── Combination helpers (not registered as templates) ─────────────────────

_EMOTION_MODIFIERS: Dict[str, str] = {
    "joy": "Infuse the delivery with bright, beaming joy and warmth.",
    "sadness": "Carry deep, quiet sadness. Voice lower, pace slower, heavy with grief.",
    "anger": "Deliver with controlled, simmering anger. Clipped, forceful, sharp.",
    "fear": "Speak with trembling fear. Voice shakes, pitch rises, pace quickens nervously.",
    "surprise": "React with genuine surprise. Sharp pitch rise, breathless disbelief.",
    "tenderness": "Speak with gentle tenderness. Soft, intimate, barely above a whisper.",
    "determination": "Speak with iron determination. Firm, steady, building conviction.",
    "despair": "Speak with hollow despair. Empty, flat, drained of hope.",
    "awe": "Speak with breathless awe. Reverent near-whisper, slow and humbled.",
    "contempt": "Drip with cold contempt. Slow, mocking, dismissive.",
    "urgency": "Deliver with pressing urgency. Fast, breathless, every second counts.",
    "calm": "Speak with serene calm. Even, unhurried, grounded and peaceful.",
    "excitement": "Buzz with contained excitement. Quick pace, rising energy, barely contained.",
    "weariness": "Speak with bone-deep weariness. Slow, heavy, each word an effort.",
    "defiance": "Deliver with fierce defiance. Sharp, unyielding, refusing to break.",
    "longing": "Speak with aching longing. Slow, yearning, voice reaching for something distant.",
    "playfulness": "Speak with light playfulness. Bouncy rhythm, teasing inflections, a smile in the voice.",
    "menace": "Deliver with quiet menace. Low, soft, dangerously calm.",
    "reverence": "Speak with deep reverence. Hushed, respectful, as if in a sacred place.",
    "bitterness": "Speak with sharp bitterness. Hard consonants, clipped delivery, acid undertones.",
}

_PACE_MODIFIERS: Dict[str, str] = {
    "very-slow": "Very slow, meditative pace with long pauses between phrases.",
    "slow": "Slow, deliberate pace allowing each word to land.",
    "moderate": "Natural, moderate pace suitable for extended listening.",
    "fast": "Quick, energetic pace driving the narrative forward.",
    "very-fast": "Rapid-fire delivery racing with urgency and breathless energy.",
    "variable": "Dynamic pace that shifts with the emotional content of the text.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_templates() -> List[VoiceTemplate]:
    """Return all registered templates."""
    return list(_TEMPLATES.values())


def get_template(template_id: str) -> Optional[VoiceTemplate]:
    """Look up a single template by its ID."""
    return _TEMPLATES.get(template_id)


def get_templates_by_category(category: str) -> List[VoiceTemplate]:
    """Return all templates in a given category."""
    return [t for t in _TEMPLATES.values() if t.category == category]


def get_categories() -> List[str]:
    """Return the unique list of template categories."""
    return sorted({t.category for t in _TEMPLATES.values()})


def get_emotion_modifiers() -> Dict[str, str]:
    """Return all available emotion modifier phrases."""
    return dict(_EMOTION_MODIFIERS)


def get_pace_modifiers() -> Dict[str, str]:
    """Return all available pace modifier phrases."""
    return dict(_PACE_MODIFIERS)


def build_instruct(
    *,
    template_id: Optional[str] = None,
    base_instruct: Optional[str] = None,
    emotion: Optional[str] = None,
    pace: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """
    Compose a full ``instruct`` string by combining a template, optional
    emotion modifier, pace modifier, and free-form extra instructions.

    Parameters
    ----------
    template_id
        If provided, the base instruct is pulled from this template.
    base_instruct
        If provided **and** template_id is ``None``, used as the raw base.
    emotion
        Key from ``_EMOTION_MODIFIERS`` (e.g. ``"joy"``, ``"anger"``).
    pace
        Key from ``_PACE_MODIFIERS`` (e.g. ``"slow"``, ``"fast"``).
    extra
        Free-form text appended at the end.

    Returns
    -------
    str
        The composed instruct string (max ~2000 chars recommended).
    """
    parts: list[str] = []

    if template_id:
        tpl = get_template(template_id)
        if tpl is None:
            raise ValueError(f"Unknown template_id: {template_id!r}")
        parts.append(tpl.instruct)
    elif base_instruct:
        parts.append(base_instruct)

    if emotion and emotion in _EMOTION_MODIFIERS:
        parts.append(_EMOTION_MODIFIERS[emotion])
    if pace and pace in _PACE_MODIFIERS:
        parts.append(_PACE_MODIFIERS[pace])
    if extra:
        parts.append(extra)

    return " ".join(parts)
