import os
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from services.vocabulary import get_connection, fetch_known_words
from services.embeddings import (
    initialize,
    active_backend,
    active_dim,
    embed,
    embed_batch,
    cosine_similarity,
)

log = logging.getLogger("vector_vocabulary")

CACHE_PATH = "cache/vocab_embeddings.npz"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass
class VocabItem:
    front: str
    back: str
    tags: list[str]
    stability: float
    embedding: np.ndarray


class VectorVocabulary:
    def __init__(self, language: str = "en"):
        self.language = language
        self._items: list[VocabItem] = []
        self._matrix: Optional[np.ndarray] = None
        self._lower_index: dict[str, int] = {}
        self._loaded = False

    def _ensure_embedder(self):
        if active_dim() == 0:
            initialize()
        return True

    def load(self, use_cache: bool = True) -> None:
        if self._loaded:
            return

        words = fetch_known_words(self.language)
        log.info("Loaded %d words from DB (language=%s)", len(words), self.language)

        items_with_emb: list[tuple] = []
        items_without_emb: list[str] = []

        for w in words:
            emb = w.get("embedding")
            if emb is not None and emb.size > 0:
                items_with_emb.append((w, emb))
            else:
                items_without_emb.append(w["front"])

        if items_without_emb:
            log.info("Computing embeddings for %d missing words...", len(items_without_emb))
            self._ensure_embedder()
            new_embs = embed_batch(items_without_emb, use_cache=use_cache)
            from services.vocabulary import update_embedding
            for i, front in enumerate(items_without_emb):
                update_embedding(front, self.language, new_embs[i])
                idx = next(j for j, w in enumerate(words) if w["front"] == front)
                words[idx]["embedding"] = new_embs[i]
            words = fetch_known_words(self.language)
            items_with_emb = []
            for w in words:
                emb = w.get("embedding")
                if emb is not None and emb.size > 0:
                    items_with_emb.append((w, emb))

        self._items = [
            VocabItem(
                front=w["front"],
                back=w["back"],
                tags=w["tags"],
                stability=w["stability"],
                embedding=emb,
            )
            for w, emb in items_with_emb
        ]
        if self._items:
            self._matrix = np.vstack([it.embedding for it in self._items])
        self._rebuild_index()
        self._loaded = True
        log.info("VectorVocabulary ready: %d items", len(self._items))

    def _rebuild_index(self) -> None:
        self._lower_index = {it.front.lower(): i for i, it in enumerate(self._items)}

    def __len__(self) -> int:
        return len(self._items)

    def words(self) -> list[str]:
        return [it.front for it in self._items]

    def contains(self, word: str) -> bool:
        return word.lower() in self._lower_index

    def tokens(self) -> set[str]:
        out: set[str] = set()
        for it in self._items:
            for tok in _WORD_RE.findall(it.front.lower()):
                out.add(tok)
        return out

    def find_similar(
        self,
        word: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[VocabItem, float]]:
        if not self._loaded:
            self.load()
        if self._matrix is None or len(self._items) == 0:
            return []
        self._ensure_embedder()
        vec = embed(word, use_cache=True).reshape(1, -1)
        sims = cosine_similarity(vec, self._matrix)[0]
        order = np.argsort(-sims)
        result: list[tuple[VocabItem, float]] = []
        for idx in order:
            score = float(sims[idx])
            if score < threshold:
                break
            result.append((self._items[int(idx)], score))
            if len(result) >= top_k:
                break
        return result

    def find_similar_batch(
        self,
        words: list[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict[str, list[tuple[VocabItem, float]]]:
        if not self._loaded:
            self.load()
        if self._matrix is None or len(self._items) == 0 or not words:
            return {}
        self._ensure_embedder()
        vecs = embed_batch(words, use_cache=True)
        sims = cosine_similarity(vecs, self._matrix)
        out: dict[str, list[tuple[VocabItem, float]]] = {}
        for i, w in enumerate(words):
            order = np.argsort(-sims[i])
            row: list[tuple[VocabItem, float]] = []
            for idx in order:
                score = float(sims[i, idx])
                if score < threshold:
                    break
                row.append((self._items[int(idx)], score))
                if len(row) >= top_k:
                    break
            out[w] = row
        return out

    def find_similar_db(
        self,
        word_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Use pgvector <-> operator for similarity search directly in DB."""
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT front, 1 - (embedding <=> %s::vector) AS similarity
                FROM cards
                WHERE language = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (word_embedding.tolist(), self.language, word_embedding.tolist(), top_k),
            )
            rows = cur.fetchall()
            cur.close()
            return [(r[0], float(r[1])) for r in rows]
        finally:
            conn.close()


_vocab_cache: dict[str, VectorVocabulary] = {}


def get_vector_vocabulary(language: str = "en") -> VectorVocabulary:
    if language not in _vocab_cache:
        _vocab_cache[language] = VectorVocabulary(language)
    return _vocab_cache[language]
