import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

LEVEL = os.getenv("LANGUAGE_LEVEL", "A2").upper()
SYSTEM_PROMPT = (
    f"You are a friendly English conversation partner. "
    f"The learner's level is CEFR {LEVEL}. "
    "Rules: reply ONLY in English. Keep it short: 1-3 sentences, 15-50 words. "
    f"Use only {LEVEL} vocabulary. Ask ONE follow-up question at the end. "
    "If the learner makes a mistake, do not correct it directly — just rephrase "
    "their idea with the correct form, then ask the next question. "
    "Be warm, encouraging. Topics: hobbies, food, family, weather, plans, work/study. "
    "Avoid politics, religion, medical/legal/financial advice, idioms, slang."
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

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]

def _chat_node(state: State) -> dict:
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment")
        
    llm = ChatGroq(
        model=model_name,
        temperature=0.7,
        max_tokens=200,
        groq_api_key=api_key,
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.get("messages") or [])
    response = llm.invoke(messages)
    return {"messages": [response]}

_graph_cache = {}

def _get_graph(chat_id: int):
    if chat_id not in _graph_cache:
        workflow = StateGraph(State)
        workflow.add_node("chat", _chat_node)
        workflow.add_edge(START, "chat")
        workflow.add_edge("chat", END)
        _graph_cache[chat_id] = workflow.compile(checkpointer=InMemorySaver())
    return _graph_cache[chat_id]

def reply(chat_id: int, user_text: str) -> str:
    """Send user input to the graph and get bot reply."""
    graph = _get_graph(chat_id)
    config = {"configurable": {"thread_id": str(chat_id)}}
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
    messages = result.get("messages") or []
    return messages[-1].content if messages else ""

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
