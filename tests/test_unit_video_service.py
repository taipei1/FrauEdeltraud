"""Unit-тесты VideoService.render_frame.

Эти тесты НЕ требуют сети: проверяют только генерацию кадра PIL.
"""
import numpy as np
import pytest

from services.video_service import VideoService

pytestmark = pytest.mark.unit


def _make_timings(words, start=0.0, dur=0.3):
    return [
        {"word": w, "start": start + i * dur, "end": start + (i + 1) * dur, "index": i}
        for i, w in enumerate(words)
    ]


def test_render_frame_no_timings_is_black():
    """Без таймингов — пустой чёрный кадр, без падения."""
    v = VideoService()
    frame = v.render_frame([])
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (v.height, v.width, 3)
    assert (frame == 0).all(), "должен быть полностью чёрным"


def test_render_frame_with_timings_has_white_pixels():
    """С таймингами — на кадре появляются белые пиксели (это и есть субтитры)."""
    v = VideoService()
    words = _make_timings(["hello", "world", "this", "is", "a", "test"])
    frame = v.render_frame(words, highlighted_index=2)

    white = (frame == 255).all(axis=2).sum()
    assert white > 100, f"на кадре должно быть много белых пикселей, нашли {white}"


def test_render_frame_highlight_draws_yellow():
    """Активное слово рисуется жёлтым (255, 255, 0)."""
    v = VideoService()
    words = _make_timings(["alpha", "beta", "gamma"])
    frame = v.render_frame(words, highlighted_index=1)
    yellow = ((frame[:, :, 0] == 255) & (frame[:, :, 1] == 255) & (frame[:, :, 2] == 0)).sum()
    assert yellow > 0, "активное слово должно быть жёлтым"


def test_render_frame_wraps_long_text():
    """Длинная строка переносится, а не обрезается за края кадра."""
    v = VideoService()
    long_text = ["word"] * 60
    words = _make_timings(long_text)
    frame = v.render_frame(words)
    assert frame.shape == (v.height, v.width, 3)
    assert (frame == 255).all(axis=2).sum() > 0


def test_render_frame_no_active_word_uses_white():
    """Если highlighted_index == -1, все слова белые."""
    v = VideoService()
    words = _make_timings(["first", "second", "third"])
    frame = v.render_frame(words, highlighted_index=-1)
    yellow = ((frame[:, :, 0] == 255) & (frame[:, :, 1] == 255) & (frame[:, :, 2] == 0)).sum()
    assert yellow == 0, "без активного слова жёлтого быть не должно"
