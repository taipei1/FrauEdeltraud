"""
Единый клиент LLM: gemini-3.5-flash → gemini-2.5-flash → Groq (llama-3.3-70b-versatile).
"""

import logging
import time
from typing import Any, List, Optional

import google.generativeai as genai
import httpx

from config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "quota",
    "rate limit",
    "resource exhausted",
    "api key not valid",
    "not found",
    "404",
    "overloaded",
    "unavailable",
    "deadline",
    "timeout",
)


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _groq_chat(prompt: str, temperature: float, json_mode: bool = False) -> str:
    if not Settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY не задан")

    payload: dict = {
        "model": Settings.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{Settings.GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {Settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"].strip()


def generate_text(
    prompt: str,
    *,
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
    """Текстовый запрос: Gemini (основная + резервная) → Groq."""
    temp = Settings.TEMPERATURE if temperature is None else temperature
    models = [Settings.GEMINI_MODEL, Settings.GEMINI_FALLBACK_MODEL]
    last_error: Optional[Exception] = None

    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            kwargs: dict = {}
            if json_mode:
                kwargs["generation_config"] = genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            elif temp is not None:
                kwargs["generation_config"] = genai.GenerationConfig(temperature=temp)

            response = model.generate_content(prompt, **kwargs)
            if model_name != models[0]:
                logger.info(f"[OK] Использована резервная Gemini: {model_name}")
            return response.text.strip()
        except Exception as e:
            last_error = e
            if _is_retryable(e):
                logger.warning(
                    f"[WARN] Gemini {model_name} недоступна ({e}); пробуем следующую..."
                )
                time.sleep(2)
                continue
            raise

    if Settings.GROQ_API_KEY:
        try:
            text = _groq_chat(prompt, temp, json_mode=json_mode)
            logger.info(f"[OK] Использована fallback-модель Groq: {Settings.GROQ_MODEL}")
            return text
        except Exception as groq_err:
            logger.error(f"[ERROR] Groq fallback failed: {groq_err}")
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Нет доступных LLM-провайдеров")


def generate_multimodal(parts: List[Any], *, temperature: Optional[float] = None) -> str:
    """Мультимодальный запрос (аудио/видео): только Gemini с переключением модели."""
    temp = Settings.TEMPERATURE if temperature is None else temperature
    models = [Settings.GEMINI_MODEL, Settings.GEMINI_FALLBACK_MODEL]
    last_error: Optional[Exception] = None
    gen_config = genai.GenerationConfig(temperature=temp) if temp is not None else None

    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            kwargs = {"generation_config": gen_config} if gen_config else {}
            response = model.generate_content(parts, **kwargs)
            if model_name != models[0]:
                logger.info(f"[OK] Использована резервная Gemini (multimodal): {model_name}")
            return response.text.strip()
        except Exception as e:
            last_error = e
            if _is_retryable(e):
                logger.warning(
                    f"[WARN] Gemini {model_name} (multimodal) недоступна ({e}); пробуем следующую..."
                )
                time.sleep(2)
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Нет доступных Gemini-моделей для multimodal")
