# =============================================================================
# core/embedder.py
# -----------------------------------------------------------------------------
# CONCEPT: Text Embeddings
# An embedding model converts a string of text into a dense vector of floats.
# Semantically similar texts produce vectors that are close in vector space.
#
#   "What is RAG?"   → [0.12, -0.34, 0.89, ...]  ← 384 numbers
#   "Explain RAG"    → [0.11, -0.33, 0.90, ...]  ← very similar!
#   "Recipe for pie" → [-0.77, 0.21, -0.55, ...] ← very different
#
# We use the 'all-MiniLM-L6-v2' model from sentence-transformers:
#   - 384-dimensional output
#   - ~80 MB download (cached after first use)
#   - Runs fully on CPU — no GPU or API key needed
# =============================================================================

from __future__ import annotations
from typing import List, Union
import numpy as np

from .chunker import Chunk


class Embedder:
    """
    Wraps a SentenceTransformer model to produce embeddings.

    Usage
    -----
        embedder = Embedder()
        vectors = embedder.embed_chunks(chunks)   # numpy array (N, 384)
        query_vec = embedder.embed_query("What is RAG?")  # shape (384,)
    """

    # Default model — small, fast, free, no API needed
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy-loaded on first use

    # ------------------------------------------------------------------
    # Lazy loading — the model is only downloaded once and cached
    # ------------------------------------------------------------------

    def _load_model(self):
        """Download and cache the embedding model on first call."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Embed a list of Chunk objects.

        Returns
        -------
        np.ndarray
            Shape (N, embedding_dim) — one row per chunk.
        """
        if not chunks:
            return np.empty((0, 384), dtype=np.float32)

        texts = [chunk.text for chunk in chunks]
        model = self._load_model()

        # encode() returns a numpy array of shape (N, dim)
        vectors = model.encode(
            texts,
            batch_size=32,           # process 32 texts at a time
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise → cosine sim = dot product
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray
            Shape (embedding_dim,)
        """
        model = self._load_model()
        vector = model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector[0].astype(np.float32)  # squeeze to 1-D

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._load_model().get_sentence_embedding_dimension()
