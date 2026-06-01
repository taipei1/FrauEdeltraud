"""Unit-тесты конфигурации и статических частей системы."""
import pytest

from config import Settings

pytestmark = pytest.mark.unit


def test_video_dimensions_divisible_by_16():
    """Размеры видео должны быть кратны 16 — требование большинства кодеков."""
    assert Settings.VIDEO_WIDTH % 16 == 0
    assert Settings.VIDEO_HEIGHT % 16 == 0


def test_tts_voice_set():
    assert Settings.TTS_VOICE
    assert "Guy" in Settings.TTS_VOICE or "Neural" in Settings.TTS_VOICE


def test_settings_singleton():
    """Settings — это контейнер статических полей, не объект с состоянием."""
    assert Settings.TEMP_DIR
    assert Settings.MODELS_DIR


def test_langgraph_workflow_compiles():
    """Workflow должен собираться без ошибок."""
    from langgraph_system.workflow import create_workflow

    graph = create_workflow()
    assert graph is not None
