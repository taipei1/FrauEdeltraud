import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("dub.stitch")


def build_ffmpeg_concat_file(segments: list[dict], output_path: Path) -> str:
    lines = ["ffconcat version 1.0"]
    for seg in segments:
        audio_path = seg["output_path"]
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            log.warning("Missing segment audio: %s", audio_path)
            continue
        duration = seg.get("end", 0) - seg.get("start", 0)
        lines.append(f"file '{audio_path.resolve()}'")
        lines.append(f"duration {duration:.3f}")

    concat_path = output_path.with_suffix(".txt")
    concat_path.write_text("\n".join(lines), encoding="utf-8")
    return str(concat_path)


def stitch_audio_segments(segments: list[dict], output_path: Path) -> Path:
    temp_dir = output_path.parent
    concat_file = temp_dir / "concat_list.txt"

    seg_paths = []
    seg_durations = []
    for seg in segments:
        ap = seg["output_path"]
        if ap.exists() and ap.stat().st_size > 0:
            seg_paths.append(str(ap.resolve()))
            seg_durations.append(seg.get("end", 0) - seg.get("start", 0))

    if not seg_paths:
        raise RuntimeError("No audio segments to stitch")

    lines = ["ffconcat version 1.0"]
    for path, dur in zip(seg_paths, seg_durations):
        escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {dur:.3f}")
    concat_file.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-safe", "0",
        "-f", "concat",
        "-i", str(concat_file),
        "-c", "copy",
        "-shortest",
        str(output_path),
    ]
    log.info("Stitching %d segments -> %s", len(seg_paths), output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        log.warning("Concat failed (%s), trying re-encode: %s", result.returncode, result.stderr[:200])
        cmd2 = [
            "ffmpeg", "-y",
            "-safe", "0",
            "-f", "concat",
            "-i", str(concat_file),
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd2, capture_output=True, timeout=300, check=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        log.info("Stitched audio: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def combine_with_video(
    dubbed_audio_path: Path,
    original_audio_path: Path,
    output_path: Path,
    video_path: Optional[Path] = None,
) -> Path:
    if video_path and video_path.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(dubbed_audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ]
        log.info("Combining dubbed audio with video -> %s", output_path.name)
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(original_audio_path),
            "-i", str(dubbed_audio_path),
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:a:0",
            "-map", "1:a:0",
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first",
            "-shortest",
            str(output_path),
        ]
        log.info("Mixing original+dubbed audio -> %s", output_path.name)

    subprocess.run(cmd, capture_output=True, timeout=600, check=True)

    if output_path.exists():
        log.info("Output: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def export_segment_map(segments: list, output_path: Path):
    data = []
    for seg in segments:
        data.append({
            "speaker": seg.get("speaker", ""),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "original": seg.get("original_text", ""),
            "translated": seg.get("translated_text", ""),
        })
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Segment map saved: %s", output_path)
