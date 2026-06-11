import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "video_dub.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("dub.main")

os.makedirs("tmp/video_dub", exist_ok=True)
os.makedirs("results/dubs", exist_ok=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Video Dubbing Pipeline")
    ap.add_argument("--url", help="YouTube URL to dub")
    ap.add_argument("--bot", action="store_true", help="Start Telegram bot")
    args = ap.parse_args()

    if args.bot:
        from video_dub.bot import start_dub_bot
        start_dub_bot()
        return 0
    elif args.url:
        from video_dub.graph import run_dub_pipeline
        result = run_dub_pipeline(args.url)
        if result.get("error"):
            log.error("Pipeline error: %s", result["error"])
            return 1
        log.info("Output: %s (%.1fs)", result.get("output_path"), result.get("elapsed", 0))
        return 0
    else:
        print("Usage:")
        print("  python -m video_dub.main --bot        # Start Telegram bot")
        print("  python -m video_dub.main --url <url>  # Dub a single video")
        return 0


if __name__ == "__main__":
    sys.exit(main())
