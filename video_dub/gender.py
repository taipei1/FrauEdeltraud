"""Gender detection by pitch (F0) analysis using Praat.

Praat (via parselmouth) extracts fundamental frequency from each speaker's
audio segments. Median F0 above 165 Hz -> female, below -> male.
"""

import logging
from pathlib import Path

log = logging.getLogger("dub.gender")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def detect_speaker_genders(
    audio_path: Path,
    segments: list,
    f0_threshold: float = 165.0,
    min_segments: int = 2,
    min_duration: float = 0.3,
    min_f0: float = 75.0,
    max_f0: float = 500.0,
) -> dict[str, str]:
    """Detect gender of each speaker by analyzing pitch (F0) of their segments.

    Args:
        audio_path: Path to audio file (WAV/MP3/OGG).
        segments: List of segments with .start, .end, .speaker attributes.
        f0_threshold: F0 in Hz above which a speaker is classified as female.
        min_segments: Minimum voiced segments per speaker to make a confident guess.
        min_duration: Skip segments shorter than this (Praat can't extract F0 reliably).
        min_f0: Praat pitch floor (typical male F0 minimum).
        max_f0: Praat pitch ceiling (typical female F0 maximum).

    Returns:
        dict mapping speaker label ("SPEAKER_00", "A", etc.) to "male", "female", or "unknown".
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        log.warning("parselmouth not installed; gender detection disabled")
        return {}

    if not audio_path.exists():
        log.warning("Audio file not found: %s", audio_path)
        return {seg.speaker: "unknown" for seg in segments}

    try:
        snd = parselmouth.Sound(str(audio_path))
    except Exception as e:
        log.warning("parselmouth failed to open %s: %s", audio_path, e)
        return {seg.speaker: "unknown" for seg in segments}
    speaker_f0s: dict[str, list[float]] = {}

    for seg in segments:
        if seg.duration < min_duration:
            continue
        try:
            excerpt = snd.extract_part(
                from_time=float(seg.start),
                to_time=float(seg.end),
                preserve_times=True,
            )
            pitch = call(excerpt, "To Pitch", 0.0, min_f0, max_f0)
            mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
            if mean_f0 and mean_f0 > 0:
                speaker_f0s.setdefault(seg.speaker, []).append(float(mean_f0))
        except Exception as e:
            log.debug("F0 extraction failed for seg %.2f-%.2f: %s", seg.start, seg.end, e)
            continue

    result: dict[str, str] = {}
    for speaker, f0s in speaker_f0s.items():
        if len(f0s) < min_segments:
            result[speaker] = "unknown"
            log.info("Speaker %s: only %d voiced segments, marking unknown", speaker, len(f0s))
            continue
        med = _median(f0s)
        gender = "female" if med > f0_threshold else "male"
        result[speaker] = gender
        log.info("Speaker %s: median F0=%.1f Hz -> %s (%d segments)",
                 speaker, med, gender, len(f0s))

    for seg in segments:
        result.setdefault(seg.speaker, "unknown")

    return result
