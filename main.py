"""Точка входа: запуск LangGraph workflow.

Использование:
  python main.py "текст для перевода"
  python main.py "https://www.youtube.com/watch?v=..."
"""
import logging
import os
import sys

proj_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from langgraph_system.workflow import run_workflow  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def main() -> int:
    logger.info("=" * 80)
    logger.info("FrauEdeltraud — Translation System")
    logger.info("=" * 80)

    if len(sys.argv) > 1:
        source = " ".join(sys.argv[1:])
    else:
        source = "Привет! Как у тебя дела? Сегодня хорошая погода."

    logger.info(f"Вход: {source[:80]}")
    result = run_workflow(source)

    if result.get("status") == "completed":
        logger.info(result.get("final_message"))
        return 0

    logger.error(f"Ошибка: {result.get('error_message')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
