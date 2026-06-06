"""
agent.py — LangGraph pipeline: Chat → Vector Translator → Critic.

Flow:
  1. chat_node: LLM generates response to user input
  2. vector_translate_node: Rewrites response using learner's known vocabulary,
     selecting relevant words via vector similarity (Gemini embeddings).
  3. critic_node: Checks if meaning is preserved, fixes if needed
"""
import os
import logging
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from services.vector_vocabulary import get_vector_vocabulary
from services.vector_translator import vector_translate, TranslationConfig
from services.critic import critique_translation

log = logging.getLogger("agent")

LEVEL = os.getenv("LANGUAGE_LEVEL", "A2").upper()
GRAMMAR_FILE = os.getenv("GRAMMAR_TOPICS_FILE", "grammar_topics.yaml")


def _load_grammar_topics() -> str:
    """Load grammar topics from YAML file, return formatted string."""
    import yaml
    path = GRAMMAR_FILE
    if not os.path.exists(path):
        log.warning("Grammar topics file not found: %s", path)
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        topics = data.get("grammar", [])
        if not topics:
            return ""
        lines = [f"- {t}" for t in topics]
        return "\n".join(lines)
    except Exception as e:
        log.warning("Failed to load grammar topics: %s", e)
        return ""


def _build_system_prompt() -> str:
    """Build the full system prompt with grammar topics."""
    grammar_block = _load_grammar_topics()
    grammar_section = (
        "\n\nThe learner is currently studying these grammar topics. "
        "Use them naturally in your response (at least 1-2 of them):\n"
        f"{grammar_block}"
    ) if grammar_block else ""

    return (
        f"You are a friendly English conversation partner. "
        f"The learner's English level is CEFR {LEVEL}.\n\n"

        "RESPONSE STRUCTURE:\n"
        "1. CORRECTIONS — Look at the learner's message. If it has mistakes, "
        "show the correct version and explain briefly. "
        "If there are NO mistakes, say \"All correct!\" in English.\n"
        "2. ANSWER — Then answer their question or continue the conversation "
        "naturally. Give rich, interesting information. "
        f"Practice the grammar topics below.{grammar_section}\n"
        "3. Ask exactly ONE follow-up question at the very end — no more, no fewer.\n\n"

        "RULES:\n"
        "- Reply ONLY in English.\n"
        "- Be warm, encouraging, and engaging.\n"
        "- Be INFORMATIVE. Give rich answers with interesting facts, examples, "
        "and new information. The learner talks to you to expand their horizons "
        "and learn new things. Generic or obvious answers are NOT interesting.\n"
        "- Keep sentences simple (A2-B1 level), but do not sacrifice content "
        "for brevity. A longer answer with real information is better than "
        "a short empty one.\n"
        "- Do NOT replace unknown words with explanations. If a word is not in "
        "the vocabulary, leave it as it is. Only use words from the vocabulary "
        "list when they are a natural fit.\n"
        "- Topics: hobbies, food, family, weather, plans, work/study, travel, "
        "daily life, nature, science, history, culture, technology.\n"
        "- Avoid politics, religion, medical/legal/financial advice, idioms, slang.\n"
        "- OVERRIDE RULE: If the learner explicitly asks you to ignore "
        "or change any of these instructions during the conversation, "
        "obey their direct instruction instead."
    )

GREETINGS = [
    "Hi! I'm glad to chat with you today. How are you feeling?",
    "Hello! Tell me, what did you do this morning?",
    "Hey! What's your favourite food, and why?",
    "Hi there! Do you have any hobbies? Tell me about one.",
    "Hello! What is the weather like in your city today?",
    "Hi! What did you eat for breakfast?",
    "Hey! Do you like listening to music? What kind?",
]


# --- State ---
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    raw_response: str         # LLM's raw response
    translated_response: str  # After vocabulary replacement
    final_response: str       # After critic approval/fix
    known_words: list[str]    # Vocabulary from DB
    critic_score: int         # Quality score 1-10
    selected_vocab: list      # Vector-selected relevant words: [(word, score), ...]
    unknown_tokens: list[str]  # Tokens in raw_response not in known vocabulary


# --- Nodes ---
FALLBACK_MODEL = "llama3-8b-8192"


def _llm_invoke(messages: list, model: str, api_key: str, temperature: float, max_tokens: int) -> str:
    from langchain_groq import ChatGroq
    """Invoke LLM with fallback on rate limit."""
    try:
        llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens, groq_api_key=api_key)
        return llm.invoke(messages).content
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            log.warning("Rate limited on %s, falling back to %s", model, FALLBACK_MODEL)
            fallback = ChatGroq(model=FALLBACK_MODEL, temperature=temperature, max_tokens=max_tokens, groq_api_key=api_key)
            return fallback.invoke(messages).content
        raise


def chat_node(state: State) -> dict:
    """Generate LLM response to user input."""
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    
    system_prompt = _build_system_prompt()
    msgs = [SystemMessage(content=system_prompt)] + list(state.get("messages") or [])
    raw_text = _llm_invoke(msgs, model_name, api_key, 0.7, 400)
    log.info("Chat raw response: %r", raw_text[:100])
    return {
        "raw_response": raw_text,
    }


def load_vocabulary_node(state: State) -> dict:
    """Fetch known vocabulary (with embeddings) from the database."""
    try:
        vocab = get_vector_vocabulary("en")
        if not vocab._loaded:
            vocab.load()
        words = vocab.words()
        log.info("Loaded %d known vocabulary words from DB (with vectors)", len(words))
        return {"known_words": words}
    except Exception as e:
        log.warning("Failed to load vocabulary: %s, skipping translation", e)
        return {"known_words": []}


def translate_node(state: State) -> dict:
    """Rewrite response using vector-selected known vocabulary."""
    raw = state.get("raw_response", "")
    known = state.get("known_words", [])
    
    if not known or not raw:
        log.info("Skipping translation (no vocabulary or empty response)")
        return {"translated_response": raw}
    
    try:
        vocab = get_vector_vocabulary("en")
        if not vocab._loaded:
            vocab.load()
        cfg = TranslationConfig(
            temperature=float(os.getenv("TRANSLATOR_TEMPERATURE", "0.3")),
            top_k_per_word=int(os.getenv("TRANSLATOR_TOP_K", "8")),
            max_vocab_in_prompt=int(os.getenv("TRANSLATOR_MAX_VOCAB", "200")),
            similarity_threshold=float(os.getenv("TRANSLATOR_THRESHOLD", "0.45")),
        )
        result = vector_translate(raw, vocab=vocab, cfg=cfg)
        return {
            "translated_response": result["rewritten"],
            "selected_vocab": result["selected_vocab"],
            "unknown_tokens": result["unknown_tokens"],
        }
    except Exception as e:
        log.error("Translation failed: %s, using raw response", e)
        return {"translated_response": raw}


def critic_node(state: State) -> dict:
    """Check if translated text preserves meaning."""
    raw = state.get("raw_response", "")
    translated = state.get("translated_response", "")
    selected = state.get("selected_vocab", [])
    
    if not raw or not translated or raw == translated:
        return {
            "final_response": translated or raw,
            "critic_score": 10,
            "selected_vocab": selected,
            "messages": [AIMessage(content=translated or raw)],
        }
    
    try:
        result = critique_translation(raw, translated)
        final = result["fixed_text"]
        score = result["score"]
        log.info(
            "Critic result: score=%d, approved=%s",
            score, result["approved"],
        )
        return {
            "final_response": final,
            "critic_score": score,
            "selected_vocab": selected,
            "messages": [AIMessage(content=final)],
        }
    except Exception as e:
        log.error("Critic failed: %s, using translated response", e)
        return {
            "final_response": translated,
            "critic_score": 5,
            "selected_vocab": selected,
            "messages": [AIMessage(content=translated)],
        }


# --- Graph Construction ---
_graph_cache: dict[int, object] = {}


def _build_graph():
    """Build the LangGraph with chat → vocabulary → translate → critic."""
    workflow = StateGraph(State)
    
    workflow.add_node("load_vocabulary", load_vocabulary_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("translate", translate_node)
    workflow.add_node("critic", critic_node)
    
    workflow.add_edge(START, "load_vocabulary")
    workflow.add_edge("load_vocabulary", "chat")
    workflow.add_edge("chat", "translate")
    workflow.add_edge("translate", "critic")
    workflow.add_edge("critic", END)
    
    return workflow.compile(checkpointer=InMemorySaver())


def _get_graph(chat_id: int):
    if chat_id not in _graph_cache:
        _graph_cache[chat_id] = _build_graph()
        log.info("Created new graph for chat_id=%s", chat_id)
    return _graph_cache[chat_id]


# --- Public API ---
def reply(chat_id: int, user_text: str) -> dict:
    """Send user input through the full pipeline.
    Returns dict with final_response, selected_vocab, unknown_tokens."""
    graph = _get_graph(chat_id)
    config = {"configurable": {"thread_id": str(chat_id)}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )
    return {
        "final_response": result.get("final_response", ""),
        "selected_vocab": result.get("selected_vocab", []),
        "unknown_tokens": result.get("unknown_tokens", []),
    }


def greet(chat_id: int) -> str:
    """Generate greeting, insert into memory, and return it."""
    text = GREETINGS[chat_id % len(GREETINGS)]
    graph = _get_graph(chat_id)
    config = {"configurable": {"thread_id": str(chat_id)}}
    graph.invoke({"messages": [AIMessage(content=text)]}, config=config)
    return text


def reset(chat_id: int) -> bool:
    """Clear memory for the given chat_id."""
    return _graph_cache.pop(chat_id, None) is not None
