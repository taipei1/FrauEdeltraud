import os
import logging
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

log = logging.getLogger("vocabulary")


def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment")
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


def fetch_known_words(language: str = "en") -> list[dict]:
    """Fetch all known vocabulary words from the cards table,
    including embedding vectors stored in pgvector."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT front, back, tags, stability, embedding
            FROM cards
            WHERE language = %s
            ORDER BY stability DESC
            """,
            (language,),
        )
        rows = cur.fetchall()
        cur.close()
        result = []
        for row in rows:
            emb = row[4]
            result.append({
                "front": row[0],
                "back": row[1],
                "tags": row[2] or [],
                "stability": row[3] or 0.0,
                "embedding": np.asarray(emb, dtype=np.float32) if emb is not None else None,
            })
        return result
    finally:
        conn.close()


def get_vocabulary_list(language: str = "en") -> list[str]:
    words = fetch_known_words(language)
    return [w["front"] for w in words]


def count_missing_embeddings(language: str = "en") -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM cards WHERE language = %s AND embedding IS NULL",
            (language,),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


def update_embedding(card_front: str, language: str, embedding: np.ndarray):
    """Store a single embedding vector for a card."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE cards SET embedding = %s WHERE front = %s AND language = %s",
            (embedding.tolist(), card_front, language),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def reindex_embeddings(language: str = "en") -> dict:
    """Compute and store embeddings for all cards where embedding IS NULL.
    Uses batch embedding for efficiency. Returns stats dict."""
    from services.embeddings import embed, embed_batch, initialize, active_dim
    initialize()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT front FROM cards WHERE language = %s AND embedding IS NULL",
            (language,),
        )
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return {"processed": 0, "total": 0}

        fronts = [r[0] for r in rows]
        log.info("Reindexing %d cards (batch embedding via %s)...",
                 len(fronts), os.getenv("EMBEDDING_BACKEND", "gemini"))

        # Batch embed all at once
        embeddings = embed_batch(fronts, use_cache=True)

        # Store each embedding
        for i, front in enumerate(fronts):
            cur = conn.cursor()
            vec = embeddings[i]
            if vec.shape[0] != active_dim():
                log.error("Dimension mismatch for %s: expected %d, got %d",
                          front, active_dim(), vec.shape[0])
                continue
            cur.execute(
                "UPDATE cards SET embedding = %s WHERE front = %s AND language = %s",
                (vec.tolist(), front, language),
            )
            conn.commit()
            cur.close()

        return {"processed": len(fronts), "total": len(fronts)}
    finally:
        conn.close()


def get_recent_words(language: str = "en", days: int = 7, limit: int = 50) -> list[str]:
    """Fetch recently added vocabulary words."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT front FROM cards
            WHERE language = %s
              AND created_at >= NOW() - INTERVAL '%s days'
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (language, str(days), limit),
        )
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]
    finally:
        conn.close()
