"""Tests for the video dubbing LangGraph pipeline with mocked dependencies."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from video_dub.graph import (
    DubState,
    download_node,
    transcribe_node,
    translate_node,
    voice_assign_node,
    synthesize_node,
    stitch_node,
    build_dub_graph,
    run_dub_pipeline,
)
from video_dub.config import DubConfig
from video_dub.download import is_youtube_url, download_audio, get_video_info
from video_dub.transcribe import SpeakerSegment, transcribe, split_long_segments, merge_short_segments
from video_dub.translate import SegmentTranslation, translate_segment
from video_dub.synthesize import assign_voices, synthesize_all, _sanitize_text, VOICE_ASSIGNMENT_CACHE


@pytest.fixture(autouse=True)
def clear_caches():
    VOICE_ASSIGNMENT_CACHE.clear()
    yield


class TestDownload:
    def test_is_youtube_url_valid(self):
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ")

    def test_is_youtube_url_invalid(self):
        assert not is_youtube_url("https://vimeo.com/12345")
        assert not is_youtube_url("not a url")
        assert not is_youtube_url("")
        assert not is_youtube_url("https://youtube.com/watch?v=short")

    def test_download_node_success(self):
        state = DubState(url="https://youtube.com/watch?v=dQw4w9WgXcQ")
        real_tmp = tempfile.mkdtemp()
        try:
            audio_path = Path(real_tmp) / "test_dub.mp3"
            audio_path.touch()
            with patch("video_dub.graph.get_video_info") as mock_info, \
                 patch("video_dub.graph.download_audio") as mock_dl, \
                 patch("video_dub.graph.tempfile.mkdtemp") as mock_tmp:
                mock_info.return_value = {"id": "dQw4w9WgXcQ", "title": "Test", "duration": 120}
                mock_dl.return_value = audio_path
                mock_tmp.return_value = real_tmp

                result = download_node(state)
                assert "error" not in result
                assert result["video_info"]["id"] == "dQw4w9WgXcQ"
                assert "audio_path" in result
                assert "work_dir" in result
        finally:
            import shutil
            shutil.rmtree(real_tmp, ignore_errors=True)

    def test_download_node_no_url(self):
        state = DubState(url="")
        result = download_node(state)
        assert "error" in result

    def test_download_node_invalid_url(self):
        state = DubState(url="https://vimeo.com/123")
        result = download_node(state)
        assert "error" in result

    def test_download_node_too_long(self):
        state = DubState(url="https://youtube.com/watch?v=dQw4w9WgXcQ")
        real_tmp = tempfile.mkdtemp()
        try:
            with patch("video_dub.graph.get_video_info") as mock_info, \
                 patch("video_dub.graph.tempfile.mkdtemp") as mock_tmp:
                mock_info.return_value = {"duration": 3600 * 2}  # 2 hours
                mock_tmp.return_value = real_tmp
                result = download_node(state)
                assert "error" in result
                assert "too long" in result["error"].lower()
        finally:
            import shutil
            shutil.rmtree(real_tmp, ignore_errors=True)

    def test_download_node_exception(self):
        state = DubState(url="https://youtube.com/watch?v=dQw4w9WgXcQ")
        real_tmp = tempfile.mkdtemp()
        try:
            with patch("video_dub.graph.get_video_info", side_effect=RuntimeError("Network error")), \
                 patch("video_dub.graph.tempfile.mkdtemp") as mock_tmp, \
                 patch("shutil.rmtree"):
                mock_tmp.return_value = real_tmp
                result = download_node(state)
                assert "error" in result
        finally:
            import shutil
            shutil.rmtree(real_tmp, ignore_errors=True)


class TestTranscribe:
    def test_transcribe_node_success(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name
        try:
            state = DubState(audio_path=audio_path)
            with patch("video_dub.graph.transcribe") as mock_t:
                mock_t.return_value = [
                    SpeakerSegment("Hello world", 0.0, 2.0, "SPEAKER_00"),
                    SpeakerSegment("How are you", 2.5, 4.0, "SPEAKER_01"),
                ]
                with patch("video_dub.graph.split_long_segments", side_effect=lambda *args, **kw: args[0]), \
                     patch("video_dub.graph.merge_short_segments", side_effect=lambda *args, **kw: args[0]):
                    result = transcribe_node(state)
                    assert "error" not in result
                    assert len(result["segments"]) == 2
                    assert result["segments"][0]["speaker"] == "SPEAKER_00"
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_node_no_audio(self):
        state = DubState(audio_path="/nonexistent/file.mp3")
        result = transcribe_node(state)
        assert "error" in result

    def test_transcribe_node_exception(self):
        state = DubState(audio_path="/tmp/fake.mp3")
        with patch("pathlib.Path.exists", return_value=True), \
             patch("video_dub.graph.transcribe", side_effect=RuntimeError("STT failed")):
            result = transcribe_node(state)
            assert "error" in result

    def test_speaker_segment_to_dict(self):
        seg = SpeakerSegment("Hello", 0.0, 1.5, "SPEAKER_00")
        d = seg.to_dict()
        assert d["text"] == "Hello"
        assert d["start"] == 0.0
        assert d["end"] == 1.5
        assert d["speaker"] == "SPEAKER_00"

    def test_split_long_segments(self):
        segs = [SpeakerSegment("word " * 100, 0.0, 20.0, "SPEAKER_00")]
        result = split_long_segments(segs, max_duration=10.0)
        assert len(result) >= 2

    def test_split_short_segments(self):
        segs = [SpeakerSegment("Hello", 0.0, 2.0, "SPEAKER_00")]
        result = split_long_segments(segs, max_duration=15.0)
        assert len(result) == 1

    def test_merge_short_segments_same_speaker(self):
        segs = [
            SpeakerSegment("Hello", 0.0, 0.5, "SPEAKER_00"),
            SpeakerSegment("world", 0.6, 1.0, "SPEAKER_00"),
        ]
        result = merge_short_segments(segs, min_duration=2.0)
        assert len(result) < len(segs)
        assert "Hello" in result[0].text

    def test_merge_short_segments_different_speakers(self):
        segs = [
            SpeakerSegment("Hello", 0.0, 0.5, "SPEAKER_00"),
            SpeakerSegment("Hi", 0.6, 1.0, "SPEAKER_01"),
        ]
        result = merge_short_segments(segs, min_duration=2.0)
        assert len(result) <= len(segs)

    def test_merge_empty(self):
        assert merge_short_segments([]) == []


class TestDiarizeAssemblyAI:
    def test_handles_string_speaker_labels(self, monkeypatch, tmp_path):
        from video_dub import transcribe as tr

        class _FakeUtt:
            def __init__(self, text, start, end, speaker):
                self.text = text
                self.start = start
                self.end = end
                self.speaker = speaker

        class _FakeTranscript:
            status = "completed"
            error = None
            utterances = [
                _FakeUtt("Hello", 0, 1000, "A"),
                _FakeUtt("World", 1000, 2000, "B"),
            ]

        class _FakeTranscriber:
            def transcribe(self, path, config):
                return _FakeTranscript()

        class _FakeSettings:
            api_key = None

        class _FakeAai:
            TranscriptStatus = type("S", (), {"error": "error"})
            settings = _FakeSettings()
            Transcriber = _FakeTranscriber
            TranscriptionConfig = lambda *a, **kw: kw

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "assemblyai", _FakeAai())
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_key")

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 100)

        result = tr.diarize_assemblyai(audio, "en")

        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_A"
        assert result[1].speaker == "SPEAKER_B"

    def test_handles_int_speaker_labels(self, monkeypatch, tmp_path):
        from video_dub import transcribe as tr

        class _FakeUtt:
            def __init__(self, text, start, end, speaker):
                self.text = text
                self.start = start
                self.end = end
                self.speaker = speaker

        class _FakeTranscript:
            status = "completed"
            error = None
            utterances = [_FakeUtt("Hi", 0, 500, 1)]

        class _FakeTranscriber:
            def transcribe(self, path, config):
                return _FakeTranscript()

        class _FakeSettings:
            api_key = None

        class _FakeAai:
            TranscriptStatus = type("S", (), {"error": "error"})
            settings = _FakeSettings()
            Transcriber = _FakeTranscriber
            TranscriptionConfig = lambda *a, **kw: kw

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "assemblyai", _FakeAai())
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_key")

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 100)

        result = tr.diarize_assemblyai(audio, "en")
        assert result[0].speaker == "SPEAKER_01"

    def test_falls_back_to_groq_when_no_key(self, monkeypatch, tmp_path):
        from video_dub import transcribe as tr

        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
        monkeypatch.setattr(tr, "transcribe_groq", lambda *a, **kw: [
            SpeakerSegment("fallback", 0.0, 1.0, "SPEAKER_00")
        ])

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 100)

        result = tr.diarize_assemblyai(audio, "en")
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_00"


class TestGenderEstimation:
    def test_numeric_speaker_id_even_is_female(self):
        from video_dub.synthesize import _estimate_gender_from_speaker_id
        assert _estimate_gender_from_speaker_id("SPEAKER_00") == "female"
        assert _estimate_gender_from_speaker_id("SPEAKER_02") == "female"

    def test_numeric_speaker_id_odd_is_male(self):
        from video_dub.synthesize import _estimate_gender_from_speaker_id
        assert _estimate_gender_from_speaker_id("SPEAKER_01") == "male"
        assert _estimate_gender_from_speaker_id("SPEAKER_13") == "male"

    def test_letter_speaker_id_does_not_crash(self):
        from video_dub.synthesize import _estimate_gender_from_speaker_id
        for label in ("A", "B", "C", "D", "E", "F", "AA", "BB"):
            gender = _estimate_gender_from_speaker_id(f"SPEAKER_{label}")
            assert gender in ("male", "female"), f"unexpected gender for SPEAKER_{label}"

    def test_assign_voices_with_letter_speakers(self):
        from video_dub.synthesize import assign_voices
        from video_dub.translate import SegmentTranslation

        pool = [
            {"voice": "en-US-JennyNeural", "gender": "female", "name": "Jenny"},
            {"voice": "en-US-GuyNeural", "gender": "male", "name": "Guy"},
        ]
        segs = [
            SegmentTranslation("hi", "Привет", "SPEAKER_A", 0.0, 1.0, "en", "ru"),
            SegmentTranslation("bye", "Пока", "SPEAKER_B", 1.0, 2.0, "en", "ru"),
            SegmentTranslation("ok", "Ок", "SPEAKER_C", 2.0, 3.0, "en", "ru"),
        ]
        result = assign_voices(segs, pool)
        assert "SPEAKER_A" in result
        assert "SPEAKER_B" in result
        assert "SPEAKER_C" in result
        assert all(v in ("en-US-JennyNeural", "en-US-GuyNeural") for v in result.values())

    def test_assign_voices_uses_gender_map(self):
        from video_dub.synthesize import assign_voices
        from video_dub.translate import SegmentTranslation

        pool = [
            {"voice": "en-US-JennyNeural", "gender": "female", "name": "Jenny"},
            {"voice": "en-US-AriaNeural", "gender": "female", "name": "Aria"},
            {"voice": "en-US-GuyNeural", "gender": "male", "name": "Guy"},
        ]
        segs = [
            SegmentTranslation("a1", "A1", "SPEAKER_A", 0.0, 1.0, "en", "ru"),
            SegmentTranslation("a2", "A2", "SPEAKER_A", 1.0, 2.0, "en", "ru"),
            SegmentTranslation("b1", "B1", "SPEAKER_B", 2.0, 3.0, "en", "ru"),
            SegmentTranslation("b2", "B2", "SPEAKER_B", 3.0, 4.0, "en", "ru"),
        ]
        gender_map = {"SPEAKER_A": "male", "SPEAKER_B": "female"}
        result = assign_voices(segs, pool, gender_map=gender_map)

        assert result["SPEAKER_A"] in ("en-US-GuyNeural",)
        assert result["SPEAKER_B"] in ("en-US-JennyNeural", "en-US-AriaNeural")

    def test_assign_voices_falls_back_to_hash_for_unknown_gender(self):
        from video_dub.synthesize import assign_voices
        from video_dub.translate import SegmentTranslation

        pool = [
            {"voice": "en-US-JennyNeural", "gender": "female", "name": "Jenny"},
            {"voice": "en-US-GuyNeural", "gender": "male", "name": "Guy"},
        ]
        segs = [
            SegmentTranslation("x", "X", "SPEAKER_X", 0.0, 1.0, "en", "ru"),
        ]
        gender_map = {"SPEAKER_X": "unknown"}
        result = assign_voices(segs, pool, gender_map=gender_map)
        assert result["SPEAKER_X"] in ("en-US-JennyNeural", "en-US-GuyNeural")


class TestPitchGenderDetection:
    def test_detect_gender_handles_synthetic_audio(self, tmp_path):
        import numpy as np
        from video_dub.gender import detect_speaker_genders
        import parselmouth

        sr = 16000
        t_male = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        male_wave = 0.3 * np.sin(2 * np.pi * 120 * t_male)
        silence = np.zeros(sr // 2)
        t_female = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        female_wave = 0.3 * np.sin(2 * np.pi * 220 * t_female)
        combined = np.concatenate([male_wave, silence, female_wave])
        snd = parselmouth.Sound(combined, sampling_frequency=sr)
        audio_path = tmp_path / "synth.wav"
        snd.save(str(audio_path), "WAV")

        from video_dub.transcribe import SpeakerSegment
        segments = [
            SpeakerSegment("m1", 0.0, 1.5, "SPEAKER_M"),
            SpeakerSegment("m2", 0.5, 2.0, "SPEAKER_M"),
            SpeakerSegment("f1", 2.5, 4.0, "SPEAKER_F"),
            SpeakerSegment("f2", 3.0, 4.5, "SPEAKER_F"),
        ]

        result = detect_speaker_genders(audio_path, segments)
        assert result.get("SPEAKER_M") == "male", f"expected male, got {result.get('SPEAKER_M')}"
        assert result.get("SPEAKER_F") == "female", f"expected female, got {result.get('SPEAKER_F')}"

    def test_unknown_for_insufficient_voiced_segments(self, tmp_path):
        from video_dub.gender import detect_speaker_genders
        from video_dub.transcribe import SpeakerSegment

        result = detect_speaker_genders(
            tmp_path / "nope.wav",
            [SpeakerSegment("hi", 0.0, 0.1, "X")],
        )
        assert result.get("X") == "unknown"


class TestTranslate:
    def test_translate_node_success(self):
        state = DubState(
            segments=[
                {"text": "Hello world", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "duration": 2.0},
            ]
        )
        with patch("video_dub.graph.translate_segments") as mock_tr:
            mock_tr.return_value = [
                SegmentTranslation("Hello world", "Привет мир", "SPEAKER_00", 0.0, 2.0, "en", "ru"),
            ]
            result = translate_node(state)
            assert "error" not in result
            assert len(result["translations"]) == 1
            assert result["translations"][0]["translated_text"] == "Привет мир"

    def test_translate_node_no_segments(self):
        state = DubState(segments=[])
        result = translate_node(state)
        assert "error" in result

    def test_translate_node_exception(self):
        state = DubState(segments=[{"text": "Hi", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "duration": 1.0}])
        with patch("video_dub.graph.translate_segments", side_effect=RuntimeError("LLM failed")):
            result = translate_node(state)
            assert "error" in result

    def test_translate_segment_cached(self):
        from video_dub.translate import TRANSLATION_CACHE
        TRANSLATION_CACHE["en:ru:test"] = "тест"
        result_seg = translate_segment("test", "en", "ru", cache=True)
        assert result_seg == "тест"
        TRANSLATION_CACHE.clear()

    def test_translate_segment_empty(self):
        assert translate_segment("", "en", "ru") == ""


class TestVoiceAssign:
    def test_voice_assign_node(self):
        state = DubState(
            translations=[
                {"original_text": "Hi", "translated_text": "Привет", "speaker": "SPEAKER_00",
                 "start": 0.0, "end": 1.0, "source_lang": "en", "target_lang": "ru"},
                {"original_text": "Bye", "translated_text": "Пока", "speaker": "SPEAKER_01",
                 "start": 2.0, "end": 3.0, "source_lang": "en", "target_lang": "ru"},
            ]
        )
        result = voice_assign_node(state)
        assert "error" not in result
        assert "SPEAKER_00" in result["speaker_voices"]
        assert "SPEAKER_01" in result["speaker_voices"]
        assert result["speaker_voices"]["SPEAKER_00"] != result["speaker_voices"]["SPEAKER_01"]

    def test_voice_assign_node_no_translations(self):
        state = DubState(translations=[])
        result = voice_assign_node(state)
        assert "error" in result

    def test_assign_voices_consistent(self):
        pool = [
            {"voice": "en-US-JennyNeural", "gender": "female", "name": "Jenny"},
            {"voice": "en-US-GuyNeural", "gender": "male", "name": "Guy"},
        ]
        segs = [
            SegmentTranslation("Hi", "Hi", "SPEAKER_00", 0.0, 1.0),
            SegmentTranslation("Bye", "Bye", "SPEAKER_01", 1.0, 2.0),
        ]
        vmap = assign_voices(segs, pool)
        assert len(vmap) == 2
        assert vmap["SPEAKER_00"] in ("en-US-JennyNeural", "en-US-GuyNeural")

    def test_assign_voices_repeatable(self):
        VOICE_ASSIGNMENT_CACHE.clear()
        pool = [
            {"voice": "en-US-JennyNeural", "gender": "female", "name": "Jenny"},
            {"voice": "en-US-GuyNeural", "gender": "male", "name": "Guy"},
        ]
        segs = [SegmentTranslation("Hi", "Hi", "SPEAKER_00", 0.0, 1.0)]
        v1 = assign_voices(segs, pool)
        v2 = assign_voices(segs, pool)
        assert v1 == v2

    def test_assign_voices_no_pool_collision(self):
        VOICE_ASSIGNMENT_CACHE.clear()
        pool = [
            {"voice": "ru-RU-DariyaNeural", "gender": "female", "name": "Dasha"},
        ]
        segs = [
            SegmentTranslation("A", "A", "SPEAKER_00", 0.0, 1.0),
            SegmentTranslation("B", "B", "SPEAKER_01", 1.0, 2.0),
            SegmentTranslation("C", "C", "SPEAKER_02", 2.0, 3.0),
        ]
        vmap = assign_voices(segs, pool)
        assert all(v == "ru-RU-DariyaNeural" for v in vmap.values())


class TestSynthesize:
    def test_synthesize_node_no_translations(self):
        state = DubState(translations=[], speaker_voices={})
        result = synthesize_node(state)
        assert "error" in result

    def test_synthesize_node_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = DubState(
                translations=[
                    {"original_text": "Hi", "translated_text": "Привет", "speaker": "SPEAKER_00",
                     "start": 0.0, "end": 1.0, "source_lang": "en", "target_lang": "ru"},
                ],
                speaker_voices={"SPEAKER_00": "ru-RU-DariyaNeural"},
                work_dir=tmpdir,
            )
            with patch("video_dub.graph.synthesize_all") as mock_syn:
                mock_syn.return_value = [
                    {"idx": 0, "speaker": "SPEAKER_00", "text": "Привет", "voice": "ru-RU-DariyaNeural",
                     "start": 0.0, "end": 1.0, "output_path": Path(tmpdir) / "test.mp3"},
                ]
                result = synthesize_node(state)
                assert "error" not in result
                assert result["translations"][0]["output_path"]


class TestStitch:
    def test_stitch_node_no_segments(self):
        state = DubState(translations=[])
        result = stitch_node(state)
        assert "error" in result

    def test_stitch_node_no_audio_paths(self):
        state = DubState(
            translations=[
                {"original_text": "Hi", "translated_text": "Привет", "speaker": "SPEAKER_00",
                 "start": 0.0, "end": 1.0, "source_lang": "en", "target_lang": "ru",
                 "output_path": ""},
            ],
        )
        with patch("pathlib.Path.exists", return_value=False):
            result = stitch_node(state)
            assert "error" in result

    def test_stitch_node_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "test_seg.mp3"
            audio_file.write_text("fake audio data")
            state = DubState(
                translations=[
                    {"original_text": "Hi", "translated_text": "Привет", "speaker": "SPEAKER_00",
                     "start": 0.0, "end": 1.0, "source_lang": "en", "target_lang": "ru",
                     "output_path": str(audio_file.resolve())},
                ],
                video_info={"title": "Test Video"},
                work_dir=tmpdir,
            )
            with patch("video_dub.graph.stitch_audio_segments") as mock_stitch, \
                 patch("video_dub.graph.export_segment_map") as mock_map, \
                 patch("pathlib.Path.mkdir"):
                mock_stitch.return_value = Path(tmpdir) / "dubbed.mp3"
                result = stitch_node(state)
                assert "error" not in result
            assert "output_path" in result


class TestFullPipeline:
    def test_build_graph(self):
        graph = build_dub_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_pipeline_invalid_url(self):
        result = run_dub_pipeline("not-a-url")
        assert "error" in result

    def test_pipeline_empty_url(self):
        result = run_dub_pipeline("")
        assert "error" in result

    def test_pipeline_mocked(self):
        old_output = os.environ.get("DUB_OUTPUT_DIR")
        old_temp = os.environ.get("DUB_TEMP_DIR")
        old_max_par = os.environ.get("DUB_MAX_PARALLEL")

        real_tmp = Path(tempfile.mkdtemp())
        VALID_ID = "dQw4w9WgXcQ"
        try:
            audio_path = real_tmp / f"{VALID_ID}.mp3"
            audio_path.touch()
            out1 = real_tmp / "out1.mp3"
            out1.touch()
            out2 = real_tmp / "out2.mp3"
            out2.touch()
            output_dir = real_tmp / "output"
            output_dir.mkdir()

            os.environ["DUB_OUTPUT_DIR"] = str(output_dir)
            os.environ["DUB_TEMP_DIR"] = str(real_tmp)
            os.environ["DUB_MAX_PARALLEL"] = "1"

            with patch("video_dub.graph.get_video_info") as mock_info, \
                 patch("video_dub.graph.download_audio") as mock_dl, \
                 patch("video_dub.graph.transcribe") as mock_tr, \
                 patch("video_dub.graph.translate_segments") as mock_tl, \
                 patch("video_dub.graph.synthesize_all") as mock_syn, \
                 patch("video_dub.graph.stitch_audio_segments") as mock_st, \
                 patch("video_dub.graph.export_segment_map") as mock_map, \
                 patch("video_dub.graph.tempfile.mkdtemp") as mock_tmp, \
                 patch("shutil.rmtree"):

                mock_info.return_value = {"id": VALID_ID, "title": "Test Video", "duration": 30}
                mock_dl.return_value = audio_path
                mock_tr.return_value = [
                    SpeakerSegment("Hello world", 0.0, 2.0, "SPEAKER_00"),
                    SpeakerSegment("How are you", 2.5, 4.0, "SPEAKER_01"),
                ]
                mock_tl.return_value = [
                    SegmentTranslation("Hello world", "Привет мир", "SPEAKER_00", 0.0, 2.0, "en", "ru"),
                    SegmentTranslation("How are you", "Как дела", "SPEAKER_01", 2.5, 4.0, "en", "ru"),
                ]
                mock_syn.return_value = [
                    {"speaker": "SPEAKER_00", "text": "Привет мир", "voice": "ru-RU-DariyaNeural",
                     "start": 0.0, "end": 2.0, "output_path": out1},
                    {"speaker": "SPEAKER_01", "text": "Как дела", "voice": "ru-RU-DmitryNeural",
                     "start": 2.5, "end": 4.0, "output_path": out2},
                ]
                mock_st.side_effect = lambda segs, out: (out.touch(), out)[1]
                mock_tmp.return_value = str(real_tmp)

                from video_dub.graph import get_dub_graph, _graph_cache
                _graph_cache.pop("default", None)

                graph = get_dub_graph()
                events = list(graph.stream({
                    "url": f"https://youtube.com/watch?v={VALID_ID}",
                    "segments": [],
                    "translations": [],
                    "speaker_voices": {},
                }, {"configurable": {"thread_id": "test"}}))

                node_outputs = {}
                for event in events:
                    for node_name, data in event.items():
                        node_outputs[node_name] = data

                assert "error" not in node_outputs.get("stitch", {}), \
                    f"Stitch error: {node_outputs.get('stitch', {}).get('error')}. " \
                    f"Node outputs: {{n: list(node_outputs[n].keys()) for n in node_outputs}}"

                assert "output_path" in node_outputs.get("stitch", {})
        finally:
            import shutil
            shutil.rmtree(str(real_tmp), ignore_errors=True)
            for k in ("DUB_OUTPUT_DIR", "DUB_TEMP_DIR", "DUB_MAX_PARALLEL"):
                v = {"DUB_OUTPUT_DIR": old_output, "DUB_TEMP_DIR": old_temp, "DUB_MAX_PARALLEL": old_max_par}.get(k)
                if v:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)


class TestConfig:
    def test_default_config(self):
        cfg = DubConfig()
        assert cfg.target_language == "ru" or cfg.target_language is not None
        assert len(cfg.voice_pool) > 0

    def test_voice_pool_ru(self):
        cfg = DubConfig()
        old_lang = cfg.target_language
        cfg.target_language = "ru"
        pool = cfg.voice_pool
        assert all(v["voice"].startswith("ru-RU-") for v in pool)
        cfg.target_language = old_lang

    def test_voice_pool_en(self):
        cfg = DubConfig()
        old_lang = cfg.target_language
        cfg.target_language = "en"
        pool = cfg.voice_pool
        assert all(v["voice"].startswith("en-") for v in pool)
        cfg.target_language = old_lang

    def test_voice_pool_unknown(self):
        cfg = DubConfig()
        old_lang = cfg.target_language
        cfg.target_language = "fr"
        pool = cfg.voice_pool
        assert len(pool) > 0
        cfg.target_language = old_lang

    def test_voice_map_fallback(self):
        cfg = DubConfig()
        voices = cfg.voice_map_fallback
        assert len(voices) > 0
        assert all("Neural" in v or "Multilingual" in v for v in voices)


class TestSanitizeText:
    def test_strips_emoji(self):
        assert _sanitize_text("🎵 Мы не чужие любви 🎵") == "Мы не чужие любви"

    def test_strips_misc_symbols(self):
        assert _sanitize_text("hello ☀ world ★") == "hello world"

    def test_collapses_whitespace(self):
        assert _sanitize_text("foo   bar\t\nbaz") == "foo bar baz"

    def test_strips_control_chars(self):
        assert _sanitize_text("hi\x00there") == "hithere"

    def test_truncates_long(self):
        long_text = "a" * 6000
        out = _sanitize_text(long_text)
        assert len(out) == 5000

    def test_empty_input(self):
        assert _sanitize_text("") == ""
        assert _sanitize_text("🎵🎵🎵") == ""


class TestSynthesizeAll:
    def _seg(self, text, speaker="SPEAKER_00", start=0.0, end=1.0):
        s = MagicMock()
        s.translated_text = text
        s.speaker = speaker
        s.start = start
        s.end = end
        return s

    def test_filters_failed_jobs(self, tmp_path, monkeypatch):
        """Regression: 0-byte outputs and exceptions must NOT be returned as successful."""
        monkeypatch.setattr("video_dub.synthesize._synthesize_with_fallback",
                            lambda text, voice, cache_path, rate, pitch, fallback: (False, voice))

        segs = [self._seg("ok text", start=0.0), self._seg("also ok", start=1.0)]
        result = synthesize_all(segs, {"SPEAKER_00": "en-US-JennyNeural"}, tmp_path, max_parallel=1)
        assert result == [], f"expected no successful jobs, got {result}"

    def test_sanitizes_emoji_before_tts(self, tmp_path, monkeypatch):
        """Emoji-laden text should be stripped of emoji before being passed to the TTS engine."""
        from video_dub import synthesize as syn_mod
        seen = []

        original = syn_mod._synthesize_with_fallback

        def spy(text, voice, cache_path, rate, pitch, fallback):
            seen.append(text)
            cache_path.write_bytes(b"OK")
            return True, voice

        monkeypatch.setattr(syn_mod, "_synthesize_with_fallback", spy)

        seg = self._seg("\U0001f3b5 Hello world \U0001f3b5")
        synthesize_all([seg], {"SPEAKER_00": "en-US-JennyNeural"}, tmp_path, max_parallel=1)
        assert seen == ["Hello world"], f"emoji not stripped before TTS, got: {seen}"

    def test_drops_empty_after_sanitization(self, tmp_path, monkeypatch):
        """Text that becomes empty after sanitization should be skipped entirely."""
        from video_dub import synthesize as syn_mod
        called = []

        def spy(text, voice, cache_path, rate, pitch, fallback):
            called.append(text)
            return True, voice

        monkeypatch.setattr(syn_mod, "_synthesize_with_fallback", spy)

        seg = self._seg("\U0001f3b5\U0001f3b5\U0001f3b5")
        result = synthesize_all([seg], {"SPEAKER_00": "en-US-JennyNeural"}, tmp_path, max_parallel=1)
        assert result == [], f"empty-after-sanitize should be skipped, got {result}"
        assert called == [], "TTS should not be called for empty text"
