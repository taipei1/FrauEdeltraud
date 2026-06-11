import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage

log = logging.getLogger("dub.translate")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

TRANSLATION_CACHE: dict[str, str] = {}


@dataclass
class SegmentTranslation:
    original_text: str
    translated_text: str
    speaker: str
    start: float
    end: float
    source_lang: str = "en"
    target_lang: str = "ru"


def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        model_name = os.getenv("DUB_LLM_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        return ChatGroq(
            model=model_name,
            temperature=float(os.getenv("DUB_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("DUB_LLM_MAX_TOKENS", "300")),
            groq_api_key=api_key,
        )
    else:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        model_name = (
            os.getenv("DUB_LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-chat"
        )
        return ChatOpenAI(
            model=model_name,
            temperature=float(os.getenv("DUB_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("DUB_LLM_MAX_TOKENS", "300")),
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )


def _load_grammar_topics(target_lang: str) -> str:
    path = os.getenv("GRAMMAR_TOPICS_FILE", "grammar_topics.yaml")
    if not os.path.exists(path):
        return ""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        topics = data.get("grammar", [])
        if not topics:
            return ""
        return "\n".join(f"- {t}" for t in topics)
    except Exception:
        return ""


def _select_relevant_vocab(text: str, max_words: int = 200) -> list[str]:
    try:
        from services.vector_translator import select_relevant_vocabulary
        from services.vector_vocabulary import get_vector_vocabulary
        vocab = get_vector_vocabulary("en")
        if not vocab._loaded:
            vocab.load()
        all_words = vocab.words()
        if not all_words:
            return []
        scored = select_relevant_vocabulary(text, all_words, max_words=max_words)
        return [w for w, s in scored if s > 0.3]
    except Exception as e:
        log.warning("Vector vocab selection failed: %s", e)
        return []


def translate_segment(
    text: str,
    source_lang: str = "en",
    target_lang: str = "ru",
    cache: bool = True,
) -> str:
    if not text.strip():
        return ""

    cache_key = f"{source_lang}:{target_lang}:{text.strip()}"
    if cache and cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    tokens = _WORD_RE.findall(text)
    known_for_translation = []
    if tokens:
        known_for_translation = _select_relevant_vocab(text, max_words=400)

    known_block = ""
    if known_for_translation:
        known_block = (
            "\nKNOWN ENGLISH VOCABULARY (prefer these in translation context, all listed words):\n"
            + "\n".join(f"- {w}" for w in known_for_translation)
        )

    grammar_block = _load_grammar_topics(target_lang)

    system = f"""You are a professional translator. Translate the following speech segment from {source_lang.upper()} to {target_lang.upper()}.

RULES:
1. Translate to {target_lang.upper()} only. Output ONLY the translation.
2. Preserve the meaning, tone, and speaking style.
3. Make it sound natural in {target_lang.upper()} — like dubbing for a video.
4. Keep the translation concise (match the original length roughly).
5. Do NOT add explanations, notes, or quotes around the text.
6. If the text is a question, keep it as a question.
7. If the text contains names, keep them unchanged.
{known_block}

GRAMMAR GUIDELINES for {target_lang.upper()}:
{grammar_block}

Remember: OUTPUT ONLY THE TRANSLATION. NO EXTRA TEXT."""

    try:
        llm = _get_llm()
        msgs = [SystemMessage(content=system), HumanMessage(content=text)]
        translated = llm.invoke(msgs).content.strip()
        translated = translated.strip('"').strip("'").strip()

        if cache:
            TRANSLATION_CACHE[cache_key] = translated

        return translated
    except Exception as e:
        log.warning("Translation failed for %r: %s", text[:50], e)
        return text


def translate_segments(
    segments: list,
    source_lang: str = "en",
    target_lang: str = "ru",
    max_parallel: int = 4,
) -> list[SegmentTranslation]:
    results: list[SegmentTranslation] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {}
        for i, seg in enumerate(segments):
            future = executor.submit(translate_segment, seg.text, source_lang, target_lang)
            future_map[future] = (i, seg)

        for future in as_completed(future_map):
            i, seg = future_map[future]
            translated_text = future.result()
            results.append(SegmentTranslation(
                original_text=seg.text,
                translated_text=translated_text,
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                source_lang=source_lang,
                target_lang=target_lang,
            ))

    results.sort(key=lambda r: r.start)
    log.info("Translated %d segments (%s -> %s)", len(results), source_lang, target_lang)
    return results
