import os
import re
import shutil
import subprocess
import logging
from pathlib import Path

log = logging.getLogger("dub.download")

_YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be)/"
    r"(watch\?v=|embed/|v/|.+/|shorts/)?(?P<id>[A-Za-z0-9_-]{11})"
)

_JS_RUNTIME_HINT: str | None = None


def _detect_js_runtime() -> str | None:
    """Find a JavaScript runtime (node or deno) for yt-dlp EJS challenge solving.

    YouTube extraction without a JS runtime is deprecated and many formats
    silently fail, so we always want to pass one through if available.
    """
    global _JS_RUNTIME_HINT
    if _JS_RUNTIME_HINT is not None:
        return _JS_RUNTIME_HINT

    node = shutil.which("node")
    if node:
        _JS_RUNTIME_HINT = f"node:{node}"
        return _JS_RUNTIME_HINT

    deno = shutil.which("deno")
    if deno:
        _JS_RUNTIME_HINT = f"deno:{deno}"
        return _JS_RUNTIME_HINT

    for candidate in (
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
    ):
        if os.path.isfile(candidate):
            _JS_RUNTIME_HINT = f"node:{candidate}"
            return _JS_RUNTIME_HINT

    _JS_RUNTIME_HINT = ""
    return None


def _yt_dlp_base(work_dir: Path, video_id: str) -> list[str]:
    cmd = [
        "yt-dlp",
        "-o", str(work_dir / f"{video_id}.%(ext)s"),
        "--no-mtime",
        "--remote-components", "ejs:github",
    ]
    js_runtime = _detect_js_runtime()
    if js_runtime:
        cmd += ["--js-runtimes", js_runtime]
    return cmd


def is_youtube_url(url: str) -> bool:
    return _YOUTUBE_RE.match(url.strip()) is not None


def _has_audio_file(work_dir: Path, video_id: str) -> Path | None:
    """Return a non-empty audio file in work_dir matching video_id, or None."""
    found = _find_audio_in_dir(work_dir, video_id)
    if found is not None and found.exists() and found.stat().st_size > 0:
        return found
    return None


def download_audio(url: str, work_dir: Path) -> Path:
    url = url.strip()
    match = _YOUTUBE_RE.match(url)
    if not match:
        raise ValueError(f"Unsupported URL: {url}")

    video_id = match.group("id")
    audio_path = work_dir / f"{video_id}.mp3"

    if audio_path.exists() and audio_path.stat().st_size > 0:
        log.info("Using cached audio for %s", video_id)
        return audio_path

    cmd = _yt_dlp_base(work_dir, video_id) + [
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        url,
    ]
    log.info("Downloading audio: yt-dlp -x --audio-format mp3 %s", url)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    needs_fallback = (
        result.returncode != 0
        or _has_audio_file(work_dir, video_id) is None
    )

    if needs_fallback:
        if result.returncode != 0:
            log.warning("yt-dlp mp3 conversion failed: %s", result.stderr.strip()[:300])
        else:
            log.warning(
                "yt-dlp returned 0 but no audio file was created in %s; "
                "falling back (likely missing JS runtime / unsupported format)",
                work_dir,
            )
        audio_path = _download_with_fallback(url, work_dir, video_id)
    else:
        audio_path = _has_audio_file(work_dir, video_id) or audio_path

    if audio_path is None or not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"Audio not found in {work_dir} after download")

    log.info("Downloaded audio: %s (%.1f MB)", audio_path, audio_path.stat().st_size / 1e6)
    return audio_path


def _download_with_fallback(url: str, work_dir: Path, video_id: str) -> Path:
    """Re-run yt-dlp without forcing mp3, then convert to mp3 with ffmpeg."""
    cmd = _yt_dlp_base(work_dir, video_id) + [
        "-x",
        url,
    ]
    log.info("Fallback download: yt-dlp -x (any format) %s", url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp fallback failed: {result.stderr.strip()[:300]}")

    raw_path = _find_audio_in_dir(work_dir, video_id)
    if raw_path is None:
        raise FileNotFoundError(f"No audio file found in {work_dir}")

    mp3_path = work_dir / f"{video_id}.mp3"
    if raw_path.suffix.lower() != ".mp3":
        log.info("Converting %s -> %s via ffmpeg", raw_path.name, mp3_path.name)
        convert_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "192k",
            str(mp3_path),
        ]
        conv = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=300)
        if conv.returncode != 0 or not mp3_path.exists():
            log.warning("ffmpeg conversion failed (%s), returning raw file", conv.stderr[:200])
            return raw_path
    else:
        mp3_path = raw_path

    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        return raw_path
    return mp3_path


def _find_audio_in_dir(work_dir: Path, video_id: str) -> Path | None:
    audio_exts = {".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".aac", ".flac"}
    candidates = []
    for p in work_dir.iterdir():
        if p.is_file() and p.suffix.lower() in audio_exts and p.stat().st_size > 0:
            if video_id in p.stem or p.stem.startswith(video_id):
                candidates.append(p)
    if not candidates:
        for p in work_dir.iterdir():
            if p.is_file() and p.suffix.lower() in audio_exts and p.stat().st_size > 0:
                candidates.append(p)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_size)
    return None


def get_video_info(url: str) -> dict:
    cmd = ["yt-dlp", "--dump-json", "--no-download"]
    js_runtime = _detect_js_runtime()
    if js_runtime:
        cmd += ["--js-runtimes", js_runtime]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp info failed: {result.stderr.strip()}")

    import json
    info = json.loads(result.stdout.strip().split("\n")[0])
    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "channel": info.get("channel", ""),
        "description": info.get("description", ""),
    }


def extract_audio_segment(audio_path: Path, start_sec: float, end_sec: float, output_path: Path) -> Path:
    duration = end_sec - start_sec
    if duration < 0.5:
        duration = 0.5
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ss", str(start_sec),
        "-t", str(duration),
        "-acodec", "libmp3lame",
        "-b:a", "128k",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    return output_path


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())
