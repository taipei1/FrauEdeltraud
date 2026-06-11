import os
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

from video_dub.config import DubConfig
from video_dub.download import download_audio, get_video_info, is_youtube_url
from video_dub.transcribe import transcribe, SpeakerSegment, split_long_segments, merge_short_segments
from video_dub.translate import translate_segments, SegmentTranslation
from video_dub.synthesize import assign_voices, synthesize_all
from video_dub.stitch import stitch_audio_segments, combine_with_video, export_segment_map

log = logging.getLogger("dub.graph")


class DubState(TypedDict, total=False):
    url: str
    video_info: dict
    audio_path: str
    segments: list
    translations: list
    speaker_voices: dict
    dubbed_audio_path: str
    output_path: str
    segment_map_path: str
    error: str
    elapsed: float
    work_dir: str


def _config() -> DubConfig:
    return DubConfig()


def download_node(state: DubState) -> dict:
    url = state.get("url", "")
    if not url:
        return {"error": "No URL provided"}

    if not is_youtube_url(url):
        return {"error": "Not a supported video URL"}

    work_dir = Path(tempfile.mkdtemp(prefix="dub_", dir=_config().temp_dir))
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_info = get_video_info(url)
        duration = video_info.get("duration", 0)
        max_dur = _config().max_duration_minutes * 60
        if duration > max_dur:
            return {"error": f"Video too long: {duration // 60} min (max {_config().max_duration_minutes} min)"}

        audio_path = download_audio(url, work_dir)
        log.info("Downloaded: %s (%.1f MB, %ds)", video_info.get("title", "?"),
                 audio_path.stat().st_size / 1e6, duration)

        return {
            "video_info": video_info,
            "audio_path": str(audio_path.resolve()),
            "work_dir": str(work_dir.resolve()),
        }
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"error": f"Download failed: {e}"}


def transcribe_node(state: DubState) -> dict:
    audio_path = state.get("audio_path", "")
    if not audio_path or not Path(audio_path).exists():
        return {"error": "Audio file not found"}

    try:
        cfg = _config()
        segments = transcribe(
            Path(audio_path),
            source_language=cfg.source_language,
            use_diarization=cfg.use_assemblyai_diarization,
        )
        segments = split_long_segments(segments, cfg.max_segment_duration)
        segments = merge_short_segments(segments, cfg.min_segment_duration)

        log.info("Transcription: %d segments, %d speakers", len(segments),
                 len({s.speaker for s in segments}))
        return {"segments": [s.to_dict() for s in segments]}
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}


def translate_node(state: DubState) -> dict:
    segments_dicts = state.get("segments", [])
    if not segments_dicts:
        return {"error": "No segments to translate"}

    segments = [SpeakerSegment(**s) for s in segments_dicts]
    cfg = _config()

    try:
        translations = translate_segments(
            segments,
            source_lang=cfg.source_language,
            target_lang=cfg.target_language,
            max_parallel=cfg.max_parallel_segments,
        )
        log.info("Translation: %d segments", len(translations))
        return {"translations": [t.__dict__ for t in translations]}
    except Exception as e:
        return {"error": f"Translation failed: {e}"}


def voice_assign_node(state: DubState) -> dict:
    translations_dicts = state.get("translations", [])
    if not translations_dicts:
        return {"error": "No translations to assign voices"}

    segments = []
    for td in translations_dicts:
        st = SegmentTranslation(**td)
        segments.append(st)

    cfg = _config()
    gender_map: dict[str, str] = {}
    audio_path = state.get("audio_path", "")
    if audio_path and Path(audio_path).exists() and len(segments) > 1:
        try:
            from video_dub.gender import detect_speaker_genders
            segment_objs = [SpeakerSegment(
                text=t.translated_text,
                start=t.start,
                end=t.end,
                speaker=t.speaker,
            ) for t in segments]
            gender_map = detect_speaker_genders(Path(audio_path), segment_objs)
            log.info("Detected genders: %s", gender_map)
        except Exception as e:
            log.warning("Gender detection failed: %s", e)
            gender_map = {}

    voice_map = assign_voices(segments, cfg.voice_pool, gender_map=gender_map)
    log.info("Voice assignment: %s", voice_map)
    return {"speaker_voices": voice_map}


def synthesize_node(state: DubState) -> dict:
    translations_dicts = state.get("translations", [])
    speaker_voices = state.get("speaker_voices", {})
    work_dir_str = state.get("work_dir", "")

    if not translations_dicts:
        return {"error": "No translations to synthesize"}

    work_dir = Path(work_dir_str) if work_dir_str else Path(_config().temp_dir)

    segments = []
    for td in translations_dicts:
        st = SegmentTranslation(**td)
        segments.append(st)

    for td in translations_dicts:
        td.pop("output_path", None)

    try:
        cfg = _config()
        jobs = synthesize_all(
            segments,
            speaker_voices,
            work_dir,
            rate=cfg.edge_tts_rate,
            pitch=cfg.edge_tts_pitch,
            max_parallel=cfg.max_parallel_segments,
            fallback_voices=cfg.voice_map_fallback,
        )

        successful_jobs = [j for j in jobs if j.get("output_path")]
        if not successful_jobs:
            return {"error": "Synthesis failed: no segments were synthesized successfully"}

        jobs_by_idx = {j["idx"]: j for j in successful_jobs}
        updated_translations = []
        for idx, td in enumerate(translations_dicts):
            if idx in jobs_by_idx:
                td = dict(td)
                td["output_path"] = str(jobs_by_idx[idx]["output_path"].resolve())
                updated_translations.append(td)
            else:
                log.warning("Dropping segment idx=%d (synth failed): %r", idx, td.get("translated_text", "")[:60])

        return {"translations": updated_translations}
    except Exception as e:
        return {"error": f"Synthesis failed: {e}"}


def stitch_node(state: DubState) -> dict:
    translations_dicts = state.get("translations", [])
    work_dir_str = state.get("work_dir", "")
    video_info = state.get("video_info", {})

    if not translations_dicts:
        return {"error": "No audio segments to stitch"}

    work_dir = Path(work_dir_str) if work_dir_str else Path(_config().temp_dir)
    output_dir = Path(_config().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title = video_info.get("title", "dubbed")
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:50]

    try:
        dubbed_audio_path = output_dir / f"{safe_title}_dubbed.mp3"
        stitch_audio_segments(
            [{"output_path": Path(t.get("output_path", "")), "start": t.get("start", 0), "end": t.get("end", 0)}
             for t in translations_dicts if t.get("output_path")],
            dubbed_audio_path,
        )

        seg_map_path = output_dir / f"{safe_title}_segments.json"
        export_segment_map(translations_dicts, seg_map_path)

        return {
            "dubbed_audio_path": str(dubbed_audio_path.resolve()),
            "output_path": str(dubbed_audio_path.resolve()),
            "segment_map_path": str(seg_map_path.resolve()),
        }
    except Exception as e:
        return {"error": f"Stitching failed: {e}"}


def build_dub_graph():
    workflow = StateGraph(DubState)

    workflow.add_node("download", download_node)
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("translate", translate_node)
    workflow.add_node("voice_assign", voice_assign_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("stitch", stitch_node)

    workflow.add_edge(START, "download")
    workflow.add_edge("download", "transcribe")
    workflow.add_edge("transcribe", "translate")
    workflow.add_edge("translate", "voice_assign")
    workflow.add_edge("voice_assign", "synthesize")
    workflow.add_edge("synthesize", "stitch")
    workflow.add_edge("stitch", END)

    graph = workflow.compile()
    return graph


_graph_cache: dict[str, object] = {}

STAGE_MAP = {
    "download": "1/6 Downloading video audio...",
    "transcribe": "2/6 Transcribing with speaker detection...",
    "translate": "3/6 Translating segments (vector translation + grammar rules)...",
    "voice_assign": "4/6 Assigning voices to speakers (multi-voice)...",
    "synthesize": "5/6 Generating speech via Edge TTS...",
    "stitch": "6/6 Stitching final audio file...",
}


def get_dub_graph():
    if "default" not in _graph_cache:
        _graph_cache["default"] = build_dub_graph()
    return _graph_cache["default"]


async def stream_dub_pipeline(url: str) -> AsyncIterator[tuple[str, str, dict]]:
    """Async generator yielding progress updates as the pipeline executes.

    Yields:
        ("progress", stage_name, human_readable_message)
        ("done", "complete", final_result_dict)
        ("error", stage_name, error_message_or_dict)
    """
    graph = get_dub_graph()
    t0 = time.time()

    initial = {
        "url": url,
        "segments": [],
        "translations": [],
        "speaker_voices": {},
    }

    final_result: dict = {}
    work_dir: str | None = None

    try:
        async for event in graph.astream(initial):
            for node_name, data in event.items():
                if node_name == "__end__":
                    final_result = dict(data) if data else final_result
                    continue

                if isinstance(data, dict) and data:
                    final_result = {**final_result, **data}

                err = data.get("error") if isinstance(data, dict) else None
                if err:
                    yield ("error", node_name, err)
                    return

                msg = STAGE_MAP.get(node_name, f"Processing: {node_name}")
                yield ("progress", node_name, msg)

                if node_name == "download" and isinstance(data, dict):
                    work_dir = data.get("work_dir", "") or work_dir

        elapsed = time.time() - t0
        final_result["elapsed"] = elapsed

        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

        err = final_result.get("error")
        if err:
            yield ("error", "pipeline", err)
        elif final_result.get("output_path"):
            yield ("done", "complete", final_result)
        else:
            yield ("error", "pipeline", "No output file produced")

    except Exception as e:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        yield ("error", "exception", str(e))


def run_dub_pipeline(url: str) -> dict:
    graph = get_dub_graph()
    t0 = time.time()

    result = graph.invoke({
        "url": url,
        "segments": [],
        "translations": [],
        "speaker_voices": {},
    })

    elapsed = time.time() - t0
    result["elapsed"] = elapsed

    work_dir = result.get("work_dir", "")
    if work_dir and elapsed > 0:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    if result.get("error"):
        log.error("Pipeline failed after %.1fs: %s", elapsed, result["error"])
    else:
        log.info("Pipeline completed in %.1fs: %s", elapsed, result.get("output_path", "?"))

    return result
