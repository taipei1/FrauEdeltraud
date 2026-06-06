"""
embeddings.py — Embedding provider abstraction.

Two backends:
  - LocalSentenceTransformers (default): offline, no API cost, 384-dim
  - GeminiEmbeddings: online via Google Gemini, 3072-dim, requires GOOGLE_API_KEY

Select via env var EMBEDDING_BACKEND=local|gemini (default: local).
Caches embeddings on disk keyed by text content.
"""
import os
import hashlib
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("embeddings")

CACHE_DIR = Path(os.getenv("EMBEDDINGS_CACHE_DIR", "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
LOCAL_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Lazy-loaded singletons
_local_model = None
_gemini_client = None
_active_backend: str | None = None
_active_dim: int | None = None


def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"emb_{digest}.pkl"


def _load_cache(key: str):
    path = _cache_path(key)
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cache(key: str, value) -> None:
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
    except Exception as e:
        log.warning("Failed to cache %s: %s", key, e)


def _get_local():
    global _local_model, _active_dim
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading local model: %s", LOCAL_MODEL_NAME)
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        _active_dim = int(_local_model.get_sentence_embedding_dimension())
        log.info("Local model loaded, dim=%d", _active_dim)
    return _local_model


def _get_gemini():
    global _gemini_client, _active_dim
    if _gemini_client is None:
        warnings.filterwarnings("ignore")
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        _gemini_client = genai
        _active_dim = 3072
        log.info("Gemini client configured, dim=%d", _active_dim)
    return _gemini_client


def initialize() -> tuple[str, int]:
    """Pick backend based on env, return (backend_name, dim)."""
    global _active_backend, _active_dim
    if BACKEND == "gemini":
        try:
            _get_gemini()
            _active_backend = "gemini"
        except Exception as e:
            log.warning("Gemini backend init failed (%s), falling back to local", e)
            _get_local()
            _active_backend = "local"
    else:
        _get_local()
        _active_backend = "local"
    return _active_backend, _active_dim or 0


def active_backend() -> str:
    global _active_backend
    if _active_backend is None:
        initialize()
    return _active_backend or "unknown"


def active_dim() -> int:
    global _active_dim
    if _active_dim is None:
        initialize()
    return _active_dim or 0


def _embed_local(text: str) -> np.ndarray:
    model = _get_local()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32)


def _embed_gemini(text: str) -> np.ndarray:
    genai = _get_gemini()
    result = genai.embed_content(model=GEMINI_MODEL, content=text)
    return np.asarray(result["embedding"], dtype=np.float32)


def embed(text: str, use_cache: bool = True) -> np.ndarray:
    """Embed a single text. Returns 1D numpy array."""
    text = (text or "").strip()
    if not text:
        return np.zeros(active_dim(), dtype=np.float32)
    if use_cache:
        cached = _load_cache(text)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
    backend = active_backend()
    try:
        if backend == "gemini":
            vec = _embed_gemini(text)
        else:
            vec = _embed_local(text)
    except Exception as e:
        log.error("Embed failed (%s) for %r: %s", backend, text[:40], e)
        vec = np.zeros(active_dim(), dtype=np.float32)
    if use_cache:
        _save_cache(text, vec.tolist())
    return vec


def embed_batch(texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
    """Embed a list of texts. Returns (N, dim) matrix."""
    backend = active_backend()
    cleaned = [(t or "").strip() for t in texts]
    dim = active_dim()
    vectors: list[np.ndarray] = [np.zeros(dim, dtype=np.float32) for _ in cleaned]

    todo_idx: list[int] = []
    todo_texts: list[str] = []
    for i, t in enumerate(cleaned):
        if not t:
            continue
        if use_cache:
            cached = _load_cache(t)
            if cached is not None:
                vectors[i] = np.asarray(cached, dtype=np.float32)
                continue
        todo_idx.append(i)
        todo_texts.append(t)

    log.info("Embedding %d/%d new texts via %s (rest cached)",
             len(todo_texts), len(cleaned), backend)

    if backend == "gemini":
        BATCH = 50
        for start in range(0, len(todo_texts), BATCH):
            batch_texts = todo_texts[start:start + BATCH]
            batch_idx = todo_idx[start:start + BATCH]
            for j, t in enumerate(batch_texts):
                try:
                    vectors[batch_idx[j]] = _embed_gemini(t)
                    if use_cache:
                        _save_cache(t, vectors[batch_idx[j]].tolist())
                except Exception as e:
                    log.warning("Batch embed failed for %r: %s", t[:40], e)
            if start + BATCH < len(todo_texts):
                time.sleep(0.5)
    else:
        if todo_texts:
            model = _get_local()
            embs = model.encode(
                todo_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )
            for j, t in enumerate(todo_texts):
                vec = np.asarray(embs[j], dtype=np.float32)
                vectors[todo_idx[j]] = vec
                if use_cache:
                    _save_cache(t, vec.tolist())

    return np.vstack(vectors)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (N, D) and rows of b (M, D).
    Returns (N, M) matrix.
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    a_n = a / a_norm
    b_n = b / b_norm
    return a_n @ b_n.T


# Backwards-compatible alias (used by older imports)
GeminiEmbeddings = None  # type: ignore
EMBEDDING_DIM = None  # set after initialize()
