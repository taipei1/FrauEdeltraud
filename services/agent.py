import os
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from services.vector_vocabulary import get_vector_vocabulary
from services.vector_translator import vector_translate, TranslationConfig, build_system_prompt
from services.critic import critique_translation
from services.vocabulary import get_recent_words

log = logging.getLogger("agent")

LEVEL = os.getenv("LANGUAGE_LEVEL", "A2").upper()
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))


def _get_llm(model: str | None = None):
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return ChatGroq(model=model_name, groq_api_key=api_key)
    else:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_text: str
    raw_response: str
    translated_response: str
    final_response: str
    known_words: list[str]
    critic_score: int
    selected_vocab: list
    unknown_tokens: list[str]
    corrections: dict
    mode: str
    story_params: dict


def _trim_history(messages: list) -> list:
    """Keep only last MAX_HISTORY_TURNS user-assistant pairs."""
    if len(messages) <= MAX_HISTORY_TURNS * 2:
        return messages
    return messages[-(MAX_HISTORY_TURNS * 2):]


def load_vocabulary_node(state: State) -> dict:
    try:
        vocab = get_vector_vocabulary("en")
        if not vocab._loaded:
            vocab.load()
        words = vocab.words()
        log.info("Loaded %d known vocabulary words", len(words))
        return {"known_words": words}
    except Exception as e:
        log.warning("Failed to load vocabulary: %s", e)
        return {"known_words": []}


def corrections_node(state: State) -> dict:
    """Analyze user's last message for mistakes and produce structured corrections."""
    user_text = state.get("user_text", "")
    if not user_text.strip():
        return {"corrections": {"correct": True, "issues": [], "corrected": ""}}

    system = """You are an English teacher. Analyze the learner's message and find mistakes.
Check: spelling, grammar, word choice, articles, prepositions, tense.

Respond with ONLY valid JSON:
{
    "correct": true/false,
    "issues": ["description of each issue"],
    "corrected": "the fully corrected version of the message"
}

If no mistakes: {"correct": true, "issues": [], "corrected": "same as original"}"""

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_text),
        ]
        raw = llm.invoke(messages).content.strip()

        import json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        log.info("Corrections: correct=%s, issues=%d", result.get("correct"), len(result.get("issues", [])))
        return {"corrections": result}
    except Exception as e:
        log.warning("Corrections node failed: %s", e)
        return {"corrections": {"correct": True, "issues": [], "corrected": user_text}}


def chat_node(state: State) -> dict:
    """Generate response using DeepSeek with vocabulary in system prompt."""
    user_text = state.get("user_text", "")
    known_words = state.get("known_words", [])
    corrections = state.get("corrections", {})
    mode = state.get("mode", "conversation")

    system_prompt = build_system_prompt(
        user_text,
        known_words,
        TranslationConfig(),
        mode=mode,
        story_params=state.get("story_params"),
    )

    # Inject corrections into the prompt
    if corrections:
        if corrections.get("correct"):
            system_prompt += "\n\nThe learner's last message was correct!"
        else:
            issues = "; ".join(corrections.get("issues", []))
            corrected = corrections.get("corrected", "")
            system_prompt += f"\n\nThe learner's last message needs correction. Issues: {issues}"
            if corrected:
                system_prompt += f"\nCorrected version: {corrected}"
            system_prompt += "\nStart your response with the corrections."

    history = _trim_history(list(state.get("messages") or []))
    msgs = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=user_text)]

    try:
        llm = _get_llm()
        raw_text = llm.invoke(msgs).content
        log.info("Chat raw output: %r", raw_text[:100])
        return {"raw_response": raw_text}
    except Exception as e:
        log.error("Chat node failed: %s", e)
        return {"raw_response": "I had trouble processing that. Could you try again?"}


def translate_node(state: State) -> dict:
    """Check that vocabulary words were used; pass through if fine."""
    raw = state.get("raw_response", "")
    known = state.get("known_words", [])

    if not raw:
        return {"translated_response": raw}

    result = vector_translate(raw, known)
    return {
        "translated_response": result["rewritten"],
        "selected_vocab": result["selected_vocab"],
        "unknown_tokens": result["unknown_tokens"],
    }


def critic_node(state: State) -> dict:
    """Check the response quality."""
    raw = state.get("raw_response", "")
    translated = state.get("translated_response", "")
    selected = state.get("selected_vocab", [])
    user_text = state.get("user_text", "")

    if not raw or not translated:
        return {
            "final_response": translated or raw,
            "critic_score": 10,
            "selected_vocab": selected,
            "messages": [AIMessage(content=translated or raw)],
        }

    try:
        result = critique_translation(raw, translated, user_message=user_text)
        final = result["fixed_text"]
        score = result["score"]

        log.info("Critic: score=%d, approved=%s", score, result["approved"])
        return {
            "final_response": final,
            "critic_score": score,
            "selected_vocab": selected,
            "messages": [AIMessage(content=final)],
        }
    except Exception as e:
        log.error("Critic failed: %s, using raw response", e)
        return {
            "final_response": translated,
            "critic_score": 5,
            "selected_vocab": selected,
            "messages": [AIMessage(content=translated)],
        }


_graph_cache: dict[int, object] = {}


def _build_graph():
    workflow = StateGraph(State)

    workflow.add_node("load_vocabulary", load_vocabulary_node)
    workflow.add_node("corrections", corrections_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("translate", translate_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "load_vocabulary")
    workflow.add_edge("load_vocabulary", "corrections")
    workflow.add_edge("corrections", "chat")
    workflow.add_edge("chat", "translate")
    workflow.add_edge("translate", "critic")
    workflow.add_edge("critic", END)

    return workflow.compile(checkpointer=InMemorySaver())


def _get_graph(chat_id: int):
    if chat_id not in _graph_cache:
        _graph_cache[chat_id] = _build_graph()
        log.info("Created new graph for chat_id=%s", chat_id)
    return _graph_cache[chat_id]


def reply(chat_id: int, user_text: str) -> dict:
    graph = _get_graph(chat_id)
    config = {"configurable": {"thread_id": str(chat_id)}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_text)],
            "user_text": user_text,
        },
        config=config,
    )
    return {
        "final_response": result.get("final_response", ""),
        "selected_vocab": result.get("selected_vocab", []),
        "unknown_tokens": result.get("unknown_tokens", []),
    }


def generate_story(chat_id: int, topic: str, word_count: int = 200) -> str:
    """Generate a story using recently added vocabulary."""
    recent = get_recent_words("en", days=7, limit=50)
    vocab = get_vector_vocabulary("en")
    if not vocab._loaded:
        vocab.load()
    known = vocab.words()

    params = {
        "topic": topic,
        "word_count": word_count,
        "recent_words": recent,
    }

    system_prompt = build_system_prompt(
        topic,
        known,
        TranslationConfig(),
        mode="story",
        story_params=params,
    )

    llm = _get_llm()
    msgs = [SystemMessage(content=system_prompt)]
    story = llm.invoke(msgs).content
    log.info("Generated story (%d words): %r", len(story.split()), story[:80])
    return story


def greet(chat_id: int) -> str:
    GREETINGS = [
        "Hi! I'm glad to chat with you today. How are you feeling?",
        "Hello! Tell me, what did you do this morning?",
        "Hey! What's your favourite food, and why?",
        "Hi there! Do you have any hobbies? Tell me about one.",
        "Hello! What is the weather like in your city today?",
        "Hi! What did you eat for breakfast?",
        "Hey! Do you like listening to music? What kind?",
    ]
    text = GREETINGS[chat_id % len(GREETINGS)]
    graph = _get_graph(chat_id)
    config = {"configurable": {"thread_id": str(chat_id)}}
    graph.invoke({"messages": [AIMessage(content=text)]}, config=config)
    return text


def reset(chat_id: int) -> bool:
    return _graph_cache.pop(chat_id, None) is not None
