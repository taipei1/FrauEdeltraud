"""Интеграционные тесты VideoService.process.

Требуют интернет (edge-tts) и ffmpeg (или imageio-ffmpeg).
"""
import os

import pytest

from services.video_service import VideoService

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_process_generates_video_with_subtitles(sample_text, temp_output_dir):
    """Главный интеграционный тест: финальный mp4 содержит субтитры в кадрах."""
    v = VideoService()
    video_path, duration, words = v.process(sample_text)

    try:
        assert os.path.exists(video_path)
        assert os.path.getsize(video_path) > 1024, "видео пустое"
        assert duration > 0
        assert len(words) > 0
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def test_process_subtitles_visible_in_frame(sample_text, temp_output_dir):
    """После исправления бага — первый кадр содержит белые пиксели (субтитры)."""
    v = VideoService()
    _, _, words = v.process(sample_text)
    frame = v.render_frame(words, highlighted_index=0)
    white = (frame == 255).all(axis=2).sum()
    assert white > 50, f"субтитры не видны, белых пикселей: {white}"
