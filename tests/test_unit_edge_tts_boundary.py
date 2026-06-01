"""Unit-тест критического бага: edge_tts WordBoundary без boundary='WordBoundary' молчит.

Это регрессионный тест: если кто-то удалит параметр boundary= из TTSService,
видео снова начнёт генерироваться без субтитров, и тест это поймает.
"""
import asyncio

import edge_tts
import pytest

pytestmark = pytest.mark.unit


def test_edge_tts_default_omits_word_boundary():
    """По умолчанию edge_tts 7.x отдаёт ТОЛЬКО SentenceBoundary."""

    async def run():
        communicate = edge_tts.Communicate(text="Hello world.", voice="en-US-GuyNeural")
        types = set()
        async for chunk in communicate.stream():
            t = chunk.get("type")
            if t:
                types.add(t)
        return types

    types = asyncio.run(run())
    assert "WordBoundary" not in types, (
        "edge-tts изменил поведение по умолчанию, "
        "TTSService нужно пересмотреть."
    )


def test_edge_tts_with_word_boundary_works():
    """С boundary='WordBoundary' приходят события по каждому слову."""

    async def run():
        communicate = edge_tts.Communicate(
            text="Hello world.",
            voice="en-US-GuyNeural",
            boundary="WordBoundary",
        )
        words = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "WordBoundary":
                words.append(chunk.get("text"))
        return words

    words = asyncio.run(run())
    assert words == ["Hello", "world"], f"unexpected words: {words}"
