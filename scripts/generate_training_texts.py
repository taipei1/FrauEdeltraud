"""
generate_training_texts.py — Generate training texts based on words from the
learner's known vocabulary database.

For each batch of N known words, the LLM is asked to write a short natural
English paragraph that uses them. The output is a JSON file with:
  - texts: list of {id, words_used, text, source} entries

Usage:
  python scripts/generate_training_texts.py --count 15 --words-per-text 6 --output results/training_texts.json
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from services.vector_vocabulary import get_vector_vocabulary

log = logging.getLogger("training_texts")


def pick_word_batches(
    vocab_words: list[str],
    n_texts: int,
    words_per_text: int,
    seed: int = 42,
) -> list[list[str]]:
    """Randomly sample N batches of words from the vocabulary."""
    rng = random.Random(seed)
    out: list[list[str]] = []
    for _ in range(n_texts):
        batch = rng.sample(vocab_words, k=min(words_per_text, len(vocab_words)))
        out.append(batch)
    return out


def generate_text(
    llm: ChatGroq,
    words: list[str],
    style: str = "A2-level conversation",
) -> str:
    """Ask the LLM to write a short text using the given words."""
    system = (
        "You are a writing assistant for English learners at CEFR A2 level. "
        "Write ONE short, natural English text (2-4 sentences, 25-60 words) "
        "that uses the given words/phrases. "
        "Rules:\n"
        " 1. Use ALL the given words naturally.\n"
        " 2. Keep it simple, conversational.\n"
        " 3. Output ONLY the text — no commentary, no title."
    )
    human = f"Words to use: {', '.join(words)}\n\nText:"
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return response.content.strip().strip('"').strip("'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12,
                    help="How many training texts to generate")
    ap.add_argument("--words-per-text", type=int, default=5,
                    help="How many known words each text should use")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/training_texts.json")
    ap.add_argument("--model", default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set in .env", file=sys.stderr)
        return 1

    print(f"Loading vocabulary...")
    vocab = get_vector_vocabulary("en")
    vocab.load()
    all_words = vocab.words()
    print(f"  loaded {len(all_words)} words")

    batches = pick_word_batches(all_words, args.count, args.words_per_text, args.seed)
    print(f"Generating {len(batches)} texts using model={args.model}...")

    llm = ChatGroq(
        model=args.model,
        temperature=0.7,
        max_tokens=300,
        groq_api_key=api_key,
    )

    results: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        try:
            text = generate_text(llm, batch)
        except Exception as e:
            log.warning("Failed to generate text %d: %s", i, e)
            continue
        if not text or len(text) < 10:
            log.warning("Skipping empty/short text %d", i)
            continue
        results.append({
            "id": i,
            "seed_words": batch,
            "text": text,
        })
        print(f"  [{i:2d}/{len(batches)}] {text[:90]}")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "count": len(results), "texts": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} texts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
