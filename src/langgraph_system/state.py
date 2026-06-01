"""Состояние LangGraph workflow.

Передаётся между агентами: extractor → translator → video_generator.
"""
from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Вход
    input_source: str

    # EXTRACTOR
    raw_text: Optional[str]
    source_language: Optional[str]
    video_path: Optional[str]
    audio_path: Optional[str]

    # TRANSLATOR
    translated_text: Optional[str]
    vocabulary_usage: dict  # {word: translation}
    vocabulary_stats: dict  # {used_count, total, percentage, used_words}

    # VIDEO GENERATOR
    video_output_path: Optional[str]
    audio_duration: Optional[float]
    telegram_message_id: Optional[int]

    # Служебные
    status: str
    error_message: Optional[str]
    final_message: Optional[str]
