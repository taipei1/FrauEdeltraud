"""LangGraph: EXTRACTOR → TRANSLATOR → VIDEO_GENERATOR → END."""
import logging

from langgraph.graph import END, StateGraph

from langgraph_system.agents import (
    agent_extractor,
    agent_translator,
    agent_video_generator,
)
from langgraph_system.state import AgentState

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("extractor", agent_extractor)
    workflow.add_node("translator", agent_translator)
    workflow.add_node("video_generator", agent_video_generator)
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "translator")
    workflow.add_edge("translator", "video_generator")
    workflow.add_edge("video_generator", END)
    return workflow.compile()


def run_workflow(input_source: str) -> AgentState:
    logger.info("=" * 80)
    logger.info(f"ЗАПУСК WORKFLOW: {input_source[:60]}...")
    logger.info("=" * 80)

    initial: AgentState = {
        "input_source": input_source,
        "status": "pending",
    }
    final = create_workflow().invoke(initial)

    logger.info("=" * 80)
    if final.get("status") == "completed":
        logger.info(f"ГОТОВО: {final.get('final_message')}")
    else:
        logger.error(f"ОШИБКА: {final.get('error_message')}")
    logger.info("=" * 80)
    return final
