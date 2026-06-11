import asyncio
import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import edge_tts

log = logging.getLogger("dub.synthesize")

TTS_CACHE_DIR = Path(os.getenv("DUB_TTS_CACHE_DIR", "cache/tts_dub"))
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VOICE_ASSIGNMENT_CACHE: dict[str, str] = {}

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\u2300-\u23FF"
    "\u2B00-\u2BFF"
    "\u2190-\u21FF"
    "]+",
    flags=re.UNICODE,
)

_MAX_TEXT_CHARS = 5000


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = "".join(
        ch for ch in cleaned
        if unicodedata.category(ch)[0] not in ("C",)
        or ch in ("\n", "\r", "\t")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > _MAX_TEXT_CHARS:
        log.warning("Truncating long TTS text: %d -> %d chars", len(cleaned), _MAX_TEXT_CHARS)
        cleaned = cleaned[:_MAX_TEXT_CHARS]
    return cleaned


def _tts_cache_path(text: str, voice: str) -> Path:
    digest = hashlib.md5(f"{voice}:{text}".encode("utf-8")).hexdigest()[:16]
    return TTS_CACHE_DIR / f"dub_{digest}.mp3"


async def _synthesize_one(text: str, voice: str, output_path: Path, rate: str = "+0%", pitch: str = "+0Hz"):
    if output_path.exists():
        if output_path.stat().st_size == 0:
            try:
                output_path.unlink()
            except OSError:
                pass
        else:
            return

    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(output_path))

    if not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        raise RuntimeError("No audio was received")


def _synthesize_sync(text: str, voice: str, output_path: Path, rate: str, pitch: str):
    asyncio.run(_synthesize_one(text, voice, output_path, rate, pitch))


def _estimate_gender_from_speaker_id(speaker: str) -> str:
    label = speaker.replace("SPEAKER_", "").strip()
    if label.isdigit():
        speaker_num = int(label)
    else:
        speaker_num = sum(ord(c) for c in label)
    return "female" if speaker_num % 2 == 0 else "male"


def assign_voices(
    segments: list,
    voice_pool: list[dict],
    gender_map: dict[str, str] | None = None,
) -> dict[str, str]:
    unique_speakers = list({seg.speaker for seg in segments})
    speaker_voices: dict[str, str] = {}

    available = list(voice_pool)

    for i, speaker in enumerate(sorted(unique_speakers)):
        if speaker in VOICE_ASSIGNMENT_CACHE:
            speaker_voices[speaker] = VOICE_ASSIGNMENT_CACHE[speaker]
            continue

        detected = (gender_map or {}).get(speaker)
        if detected in ("male", "female"):
            estimated_gender = detected
        else:
            estimated_gender = _estimate_gender_from_speaker_id(speaker)
        matching = [v for v in available if v["gender"] == estimated_gender]
        if not matching:
            matching = available

        assigned = matching[i % len(matching)]
        voice_name = assigned["voice"]
        speaker_voices[speaker] = voice_name
        VOICE_ASSIGNMENT_CACHE[speaker] = voice_name
        available.remove(assigned)
        if not available:
            available = list(voice_pool)

    log.info("Voice assignment: %s", {s: speaker_voices[s] for s in sorted(speaker_voices)})
    return speaker_voices


def _synthesize_with_retry(
    text: str, voice: str, cache_path: Path, rate: str, pitch: str, max_retries: int = 2
) -> bool:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            _synthesize_sync(text, voice, cache_path, rate, pitch)
            if cache_path.exists() and cache_path.stat().st_size > 0:
                return True
            last_err = RuntimeError("Output file is empty after synthesis")
        except Exception as e:
            last_err = e
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass
        if attempt < max_retries:
            log.warning("TTS retry %d/%d for '%s...': %s",
                        attempt + 1, max_retries, text[:30], last_err)
    return False


def _synthesize_with_fallback(
    text: str,
    voice: str,
    cache_path: Path,
    rate: str,
    pitch: str,
    fallback_voices: list[str],
) -> tuple[bool, str]:
    voices_to_try = [voice] + [v for v in fallback_voices if v != voice]
    for v in voices_to_try:
        if _synthesize_with_retry(text, v, cache_path, rate, pitch, max_retries=1):
            return True, v
    return False, voice


def synthesize_all(
    segments: list,
    speaker_voices: dict[str, str],
    work_dir: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    max_parallel: int = 4,
    fallback_voices: list[str] | None = None,
) -> list[dict]:
    jobs: list[dict] = []
    sanitized_map: dict[int, str] = {}

    fallback_voices = fallback_voices or ["en-US-JennyNeural", "en-US-AriaNeural"]

    for idx, seg in enumerate(segments):
        original = seg.translated_text
        sanitized = _sanitize_text(original)
        if not sanitized:
            log.warning("Skipping empty/sanitized text at idx=%d: %r", idx, original[:60])
            continue
        if sanitized != original:
            log.info("Sanitized text at idx=%d: %r -> %r", idx, original[:40], sanitized[:40])
        sanitized_map[idx] = sanitized

        voice = speaker_voices.get(seg.speaker, speaker_voices.get("SPEAKER_00", "en-US-JennyNeural"))
        jobs.append({
            "idx": idx,
            "speaker": seg.speaker,
            "text": sanitized,
            "original_text": original,
            "voice": voice,
            "start": seg.start,
            "end": seg.end,
            "output_path": None,
        })

    if not jobs:
        log.warning("No synthesizable segments (all text was empty after sanitization)")
        return []

    log.info("Submitting %d TTS jobs (parallel=%d)", len(jobs), max_parallel)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_job: dict = {}
        for job in jobs:
            cache_path = _tts_cache_path(job["text"], job["voice"])
            if cache_path.exists() and cache_path.stat().st_size > 0:
                job["output_path"] = cache_path
                log.info("[%d] TTS cache hit: %s", job["idx"], cache_path.name)
                continue
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            future = executor.submit(
                _synthesize_with_fallback,
                job["text"], job["voice"], cache_path, rate, pitch, fallback_voices,
            )
            future_to_job[future] = job
            job["output_path"] = cache_path

        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                ok, used_voice = future.result()
                completed += 1
                if ok:
                    if used_voice != job["voice"]:
                        log.info("[%d/%d] TTS done via fallback: %s (used %s instead of %s)",
                                 completed, len(future_to_job), job["speaker"],
                                 used_voice, job["voice"])
                        job["voice"] = used_voice
                    else:
                        log.info("[%d/%d] TTS done: %s (%.1fs)",
                                 completed, len(future_to_job), job["speaker"],
                                 job["end"] - job["start"])
                else:
                    log.error("[%d/%d] TTS FAILED (all fallbacks): %s text=%r",
                              completed, len(future_to_job), job["speaker"], job["text"][:80])
                    job["output_path"] = None
            except Exception as e:
                log.error("[%d/%d] TTS EXCEPTION: %s text=%r err=%s",
                          completed, len(future_to_job), job["speaker"], job["text"][:80], e)
                job["output_path"] = None

    successful = [j for j in jobs if j["output_path"] is not None and j["output_path"].exists()
                  and j["output_path"].stat().st_size > 0]
    failed = len(jobs) - len(successful)
    log.info("Synthesized %d/%d segments (%d failed)", len(successful), len(jobs), failed)
    return successful
