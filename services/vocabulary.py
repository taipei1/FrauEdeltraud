"""
vocabulary.py — Fetches known vocabulary from PostgreSQL and provides
vector-based word matching using cosine similarity.
"""
import os
import logging
import psycopg2
from typing import Optional

log = logging.getLogger("vocabulary")


def get_connection():
    """Create a database connection."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment")
    return psycopg2.connect(db_url)


def fetch_known_words(language: str = "en") -> list[dict]:
    """Fetch all known vocabulary words from the cards table.
    
    Returns a list of dicts with keys: front, back, tags, stability.
    Sorted by stability DESC (most stable/well-known words first).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT front, back, tags, stability
            FROM cards
            WHERE language = %s
            ORDER BY stability DESC
            """,
            (language,),
        )
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "front": row[0],
                "back": row[1],
                "tags": row[2] or [],
                "stability": row[3] or 0.0,
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_vocabulary_list(language: str = "en") -> list[str]:
    """Return a flat list of known English words/phrases."""
    words = fetch_known_words(language)
    return [w["front"] for w in words]


def build_vocabulary_prompt(words: list[str], max_words: int = 500) -> str:
    """Build a formatted vocabulary list for LLM prompts.
    
    Limits to max_words most stable (best-known) words to avoid
    exceeding context limits.
    """
    selected = words[:max_words]
    return ", ".join(selected)
