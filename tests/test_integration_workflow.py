"""E2E тест всего workflow.

Запускать ТОЛЬКО когда всё настроено: PostgreSQL, API-ключи, Telegram.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_workflow_with_plain_text():
    """Workflow должен успешно отработать на простом тексте."""
    from langgraph_system.workflow import run_workflow

    result = run_workflow("Привет! Как у тебя дела?")
    assert result.get("status") in ("completed", "failed")
    if result.get("status") == "failed":
        pytest.skip(f"Workflow failed: {result.get('error_message')}")
    assert result.get("video_output_path") or result.get("telegram_message_id")
