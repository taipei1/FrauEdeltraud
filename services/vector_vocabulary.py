"""
vector_vocabulary.py — Vocabulary loaded from the cards table, with vector
embeddings computed via the configured backend (default: sentence-transformers).
Provides semantic search for similar words.
"""
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
    front: str            # English word/phrase
    back: str             # Russian translation
    tags: list[str]
    stability: float
    embedding: np.ndarray # shape (dim,)


class VectorVocabulary:
    """In-memory vocabulary with embeddings and semantic search."""

    def __init__(self, language: str = "en"):
        self.language = language
        self._items: list[VocabItem] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim)
        self._lower_index: dict[str, int] = {}     # front.lower() -> idx
        self._embedder = None
        self._loaded = False

    def _ensure_embedder(self):
        if active_dim() == 0:
            initialize()
        return True

    def load(self, use_cache: bool = True) -> None:
        """Load cards from DB, compute or load embeddings from cache."""
        if self._loaded:
            return

        words = fetch_known_words(self.language)
        log.info("Loaded %d words from DB (language=%s)", len(words), self.language)

        if use_cache and self._try_load_cache(words):
            log.info("VectorVocabulary loaded from cache: %d items", len(self._items))
            self._loaded = True
            return

        log.info("Computing embeddings for %d words via %s...",
                 len(words), active_backend())
        self._ensure_embedder()
        fronts = [w["front"] for w in words]
        embeddings = embed_batch(fronts, use_cache=True)
        log.info("Embeddings shape: %s", embeddings.shape)

        self._items = [
            VocabItem(
                front=w["front"],
                back=w["back"],
                tags=w["tags"],
                stability=w["stability"],
                embedding=embeddings[i],
            )
            for i, w in enumerate(words)
        ]
        self._matrix = embeddings
        self._rebuild_index()
        if use_cache:
            self._save_cache()
        self._loaded = True

    def _rebuild_index(self) -> None:
        self._lower_index = {it.front.lower(): i for i, it in enumerate(self._items)}

    def _try_load_cache(self, words: list[dict]) -> bool:
        if not os.path.exists(CACHE_PATH):
            return False
        try:
            data = np.load(CACHE_PATH, allow_pickle=True)
            cached_fronts = list(data["fronts"])
            cached_backs = list(data["backs"])
            cached_tags = list(data["tags"])
            cached_stab = list(data["stabilities"])
            cached_emb = data["embeddings"]
            if len(cached_fronts) != len(words):
                log.info("Cache size mismatch (%d vs %d), will recompute",
                         len(cached_fronts), len(words))
                return False
            db_fronts = [w["front"] for w in words]
            if cached_fronts != db_fronts:
                log.info("Vocabulary changed, recomputing embeddings")
                return False
            self._items = [
                VocabItem(
                    front=cached_fronts[i],
                    back=cached_backs[i],
                    tags=list(cached_tags[i]),
                    stability=float(cached_stab[i]),
                    embedding=cached_emb[i],
                )
                for i in range(len(cached_fronts))
            ]
            self._matrix = cached_emb
            self._rebuild_index()
            return True
        except Exception as e:
            log.warning("Failed to load cache: %s", e)
            return False

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            np.savez(
                CACHE_PATH,
                fronts=np.array([it.front for it in self._items], dtype=object),
                backs=np.array([it.back for it in self._items], dtype=object),
                tags=np.array([it.tags for it in self._items], dtype=object),
                stabilities=np.array([it.stability for it in self._items], dtype=np.float32),
                embeddings=self._matrix,
            )
            log.info("Saved vector vocab cache: %s", CACHE_PATH)
        except Exception as e:
            log.warning("Failed to save cache: %s", e)

    def __len__(self) -> int:
        return len(self._items)

    def words(self) -> list[str]:
        return [it.front for it in self._items]

    def contains(self, word: str) -> bool:
        return word.lower() in self._lower_index

    def tokens(self) -> set[str]:
        """Set of all tokens in all front strings (lowercased)."""
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
        """Find top-k most similar known words to the given word/phrase."""
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
        """Vectorized: find similar known words for many unknown words at once."""
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


_vocab_cache: dict[str, VectorVocabulary] = {}


def get_vector_vocabulary(language: str = "en") -> VectorVocabulary:
    if language not in _vocab_cache:
        _vocab_cache[language] = VectorVocabulary(language)
    return _vocab_cache[language]
