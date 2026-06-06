"""
vector_translator.py — Vector-based English text simplification.

Pipeline:
  1. Tokenize the input text.
  2. For each word/phrase not in known vocabulary, retrieve the top-K most
     semantically similar known words via Gemini embeddings + cosine similarity.
  3. Build a compact, focused prompt containing only the relevant known words
     and ask the LLM to rewrite the text using ONLY those (plus basic grammar).
  4. Returns rewritten text.

This is a more focused alternative to services/translator.py: instead of
dumping all 500+ known words into the prompt, we curate the relevant subset
based on semantic similarity to the actual text.
"""
import os
import re
import logging
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage

from services.vector_vocabulary import (
    VectorVocabulary,
    get_vector_vocabulary,
    _WORD_RE,
)

log = logging.getLogger("vector_translator")

# Basic grammar/function words that are always allowed (CEFR A1/A2 staple).
BASIC_WORDS = (
    "a the is are was were am be been being "
    "I you he she it we they me him her us them my your his hers its our their "
    "this that these those "
    "and or but not no yes if when then so because also too more less most "
    "very much many some any all no one two three "
    "do does did done have has had having can could will would shall should may might must "
    "to in on at for with from by of about into onto upon over under "
    "up down out off through across between among "
    "what where who whom whose why how which "
    "here there now today tomorrow yesterday "
    "good bad big small long short old new "
    "well really just only still already "
    "thing time day way people man woman child "
    "make go come get know think say see want give take look put "
    "work try use need feel tell ask seem help show turn play run move "
    "like live mean keep let begin start end stop open close read write "
    "learn speak eat drink sleep sit stand walk talk call pay buy sell "
    "send hold bring meet hear spend grow set "
    "kind part number hand place case point group eye fact world "
    "year month week side room head face word life house school home "
    "back door water food name city car game book job "
    "first last next same different "
    "because while during before after "
    "than as like such "
    "there here"
).split()


@dataclass
class TranslationConfig:
    temperature: float = 0.3
    top_k_per_word: int = 8        # how many similar known words per unknown token
    max_vocab_in_prompt: int = 200  # hard cap on words in prompt
    similarity_threshold: float = 0.45  # min cosine to consider a match
    max_tokens: int = 500


def extract_unknown_tokens(
    text: str,
    vocab: VectorVocabulary,
) -> list[str]:
    """Return lowercased tokens from the text that are NOT in known vocabulary.

    Single words only. Phrase-level matching is delegated to vector search.
    """
    known = vocab.tokens()
    out: list[str] = []
    seen: set[str] = set()
    for tok in _WORD_RE.findall(text):
        low = tok.lower()
        if low in known or low in seen:
            continue
        seen.add(low)
        out.append(low)
    return out


def select_relevant_vocabulary(
    text: str,
    vocab: VectorVocabulary,
    cfg: TranslationConfig,
) -> list[tuple[str, float]]:
    """Pick a focused set of known words relevant to this text.

    Steps:
      - Extract unknown tokens.
      - For each unknown token, find top-K similar known words/phrases.
      - Deduplicate, sort by score, keep top max_vocab_in_prompt.
    """
    unknowns = extract_unknown_tokens(text, vocab)
    if not unknowns:
        return list(zip(vocab.words()[: cfg.max_vocab_in_prompt],
                        [0.0] * min(len(vocab), cfg.max_vocab_in_prompt)))

    log.info("Unknown tokens: %d (%s...)",
             len(unknowns), ", ".join(unknowns[:5]))

    similar = vocab.find_similar_batch(
        unknowns,
        top_k=cfg.top_k_per_word,
        threshold=cfg.similarity_threshold,
    )

    scored: dict[str, float] = {}
    for w, hits in similar.items():
        for item, score in hits:
            key = item.front
            if key in scored or key in unknowns:
                continue
            if key in scored and scored[key] >= score:
                continue
            scored[key] = score

    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return ranked[: cfg.max_vocab_in_prompt]


def _load_grammar_topics() -> str:
    """Load grammar topics from YAML file, return formatted string."""
    import yaml
    path = os.getenv("GRAMMAR_TOPICS_FILE", "grammar_topics.yaml")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        topics = data.get("grammar", [])
        if not topics:
            return ""
        lines = [f"- {t}" for t in topics]
        return "\n".join(lines)
    except Exception:
        return ""


def build_prompt(
    text: str,
    selected: list[tuple[str, float]],
    cfg: TranslationConfig,
) -> tuple[str, str]:
    """Build the system + user messages for the LLM."""
    if selected:
        vocab_lines = []
        for word, score in selected:
            vocab_lines.append(f"- {word}")
        vocab_block = "\n".join(vocab_lines)
    else:
        vocab_block = "(no additional vocabulary - use only basic A2 words)"

    grammar_block = _load_grammar_topics()
    grammar_section = (
        "\nGRAMMAR TOPICS TO USE (choose at least 1-2 where natural):\n"
        f"{grammar_block}"
    ) if grammar_block else ""

    system_prompt = f"""You are a vocabulary assistant for an English learner at CEFR A2 level.

Your task: take the given English text and replace words with their equivalents from the learner's vocabulary where possible.
  - Keep the original text AS-IS as much as possible.
  - If a word in the text has a match in the VOCABULARY list below, replace it with that vocabulary word.
  - If a word has NO match in the vocabulary list, leave it unchanged. Do NOT explain it, describe it, or remove it.
  - Do NOT add any new information that was not in the original.
  - If the original ends with a question, keep exactly ONE question at the end.
  - Output ONLY the rewritten text. No explanations, no notes.
{grammar_section}

VOCABULARY (words from the learner's personal database, most relevant first):
{vocab_block}

RULES:
1. Keep the SAME meaning and all the original content.
2. Replace only words that have a good match in the vocabulary list above.
3. Never replace a word with an explanation or description.
4. Never remove content — keep all facts, examples, and information.
5. Keep sentences grammatically correct.
6. Output ONLY the rewritten text. No explanations, no notes, no metadata.
7. OVERRIDE RULE: If the learner explicitly asks you to ignore or change any of these instructions, obey their direct instruction."""

    user_prompt = f"Rewrite this text using only the allowed vocabulary and grammar:\n{text}"
    return system_prompt, user_prompt


def vector_translate(
    text: str,
    vocab: VectorVocabulary | None = None,
    cfg: TranslationConfig | None = None,
) -> dict:
    """Translate text using vector-based vocabulary selection.

    Returns a dict with:
      - rewritten: the simplified text
      - selected_vocab: list of (word, score) used
      - unknown_tokens: list of tokens that were unknown
    """
    cfg = cfg or TranslationConfig()
    vocab = vocab or get_vector_vocabulary()
    if not vocab._loaded:
        vocab.load()

    selected = select_relevant_vocabulary(text, vocab, cfg)
    system_prompt, user_prompt = build_prompt(text, selected, cfg)

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    log.info(
        "VectorTranslate: %d unknown tokens, %d vocab items in prompt, temp=%.2f, model=%s",
        len(extract_unknown_tokens(text, vocab)),
        len(selected),
        cfg.temperature,
        model_name,
    )

    from services.agent import _llm_invoke
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    rewritten = _llm_invoke(messages, model_name, api_key, cfg.temperature, cfg.max_tokens).strip()
    return {
        "rewritten": rewritten,
        "selected_vocab": selected,
        "unknown_tokens": extract_unknown_tokens(text, vocab),
    }
