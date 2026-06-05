import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

from services.bot import start_telegram_bot

def main() -> int:
    try:
        start_telegram_bot()
        return 0
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
        return 0
    except Exception as e:
        log.critical("Failed to start bot: %s", e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
