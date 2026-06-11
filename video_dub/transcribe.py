import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("dub.transcribe")


class SpeakerSegment:
    def __init__(self, text: str, start: float, end: float, speaker: str = "SPEAKER_00", **kwargs):
        self.text = text
        self.start = start
        self.end = end
        self.speaker = speaker
        self.duration = end - start

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "duration": self.duration,
        }

    def __repr__(self):
        return f"[{self.start:.1f}-{self.end:.1f}] {self.speaker}: {self.text[:50]}"


def transcribe_groq(audio_path: Path, language: str = "en", model: str = "whisper-large-v3") -> list[SpeakerSegment]:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        suffix = audio_path.suffix.lower()
        mime_map = {".mp3": "audio/mp3", ".ogg": "audio/ogg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
        mime = mime_map.get(suffix, "application/octet-stream")

        result = client.audio.transcriptions.create(
            file=(audio_path.name, f, mime),
            model=model,
            language=language,
            response_format="verbose_json",
            temperature=0.0,
        )

    segments = []
    raw_segments = getattr(result, "segments", None) or []
    for seg in raw_segments:
        if isinstance(seg, dict):
            text = (seg.get("text") or "").strip()
            start = float(seg.get("start") or 0)
            end = float(seg.get("end") or 0)
        else:
            text = (getattr(seg, "text", "") or "").strip()
            start = float(getattr(seg, "start", 0) or 0)
            end = float(getattr(seg, "end", 0) or 0)
        if text:
            segments.append(SpeakerSegment(text, start, end, "SPEAKER_00"))

    if not segments:
        full_text = result.text if isinstance(result, str) else getattr(result, "text", "")
        if full_text and full_text.strip():
            segments.append(SpeakerSegment(full_text.strip(), 0.0, 0.0, "SPEAKER_00"))

    log.info("Transcribed %d segments (%.1fs total audio)", len(segments),
             sum(s.duration for s in segments))
    return segments


def diarize_assemblyai(audio_path: Path, language: str = "en") -> list[SpeakerSegment]:
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        log.warning("ASSEMBLYAI_API_KEY not set, falling back to Groq transcription")
        return transcribe_groq(audio_path, language)

    import assemblyai as aai
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        language_code=language,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(str(audio_path), config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI failed: {transcript.error}")

    segments = []
    for utt in transcript.utterances:
        text = (utt.text or "").strip()
        if not text:
            continue
        raw_speaker = getattr(utt, "speaker", None)
        if raw_speaker is None:
            speaker = "SPEAKER_00"
        elif isinstance(raw_speaker, int):
            speaker = f"SPEAKER_{raw_speaker:02d}"
        else:
            label = str(raw_speaker).strip()
            if label.isdigit():
                speaker = f"SPEAKER_{int(label):02d}"
            else:
                speaker = f"SPEAKER_{label}"
        segments.append(SpeakerSegment(
            text=text,
            start=utt.start / 1000.0,
            end=utt.end / 1000.0,
            speaker=speaker,
        ))

    if not segments:
        log.warning("AssemblyAI returned no utterances, falling back to Groq")
        return transcribe_groq(audio_path, language)

    log.info("Diarized %d segments across %d speakers", len(segments),
             len({s.speaker for s in segments}))
    return segments


def transcribe(audio_path: Path, source_language: str = "en", use_diarization: bool = False) -> list[SpeakerSegment]:
    if use_diarization and os.getenv("ASSEMBLYAI_API_KEY", ""):
        return diarize_assemblyai(audio_path, source_language)
    return transcribe_groq(audio_path, source_language)


def split_long_segments(segments: list[SpeakerSegment], max_duration: float = 15.0) -> list[SpeakerSegment]:
    result = []
    for seg in segments:
        if seg.duration <= max_duration:
            result.append(seg)
            continue
        text = seg.text
        words = text.split()
        n_chunks = max(2, int(seg.duration / max_duration) + 1)
        chunk_size = max(1, len(words) // n_chunks)
        chunk_duration = seg.duration / n_chunks

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            chunk_start = seg.start + i / len(words) * seg.duration
            chunk_end = chunk_start + len(chunk_words) / len(words) * seg.duration
            result.append(SpeakerSegment(chunk_text, chunk_start, chunk_end, seg.speaker))

    log.info("Split %d segments -> %d segments (max %.1fs)", len(segments), len(result), max_duration)
    return result


def merge_short_segments(segments: list[SpeakerSegment], min_duration: float = 1.0) -> list[SpeakerSegment]:
    if not segments:
        return segments

    result = []
    buffer = segments[0]

    for seg in segments[1:]:
        if buffer.speaker == seg.speaker and buffer.duration + seg.duration < min_duration * 2:
            buffer = SpeakerSegment(
                text=buffer.text + " " + seg.text,
                start=buffer.start,
                end=seg.end,
                speaker=buffer.speaker,
            )
        else:
            if buffer.duration >= min_duration or not result:
                result.append(buffer)
            else:
                result[-1] = SpeakerSegment(
                    text=result[-1].text + " " + buffer.text,
                    start=result[-1].start,
                    end=buffer.end,
                    speaker=result[-1].speaker,
                )
            buffer = seg

    if buffer.duration >= min_duration or not result:
        result.append(buffer)
    else:
        result[-1] = SpeakerSegment(
            text=result[-1].text + " " + buffer.text,
            start=result[-1].start,
            end=buffer.end,
            speaker=result[-1].speaker,
        )

    log.info("Merged %d segments -> %d segments (min %.1fs)", len(segments), len(result), min_duration)
    return result
