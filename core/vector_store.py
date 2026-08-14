# =============================================================================
# core/vector_store.py
# -----------------------------------------------------------------------------
# CONCEPT: Vector Database
# After we embed our chunks we need somewhere to store the vectors so we can
# search them quickly. A vector store does two things:
#   1. ADD    – Store (vector, metadata) pairs
#   2. SEARCH – Given a query vector, return the K most similar stored vectors
#
# We use FAISS (Facebook AI Similarity Search):
#   - IndexFlatIP  → "flat" = no compression, "IP" = Inner Product (= cosine
#                    similarity when vectors are L2-normalised)
#   - Exact search — no approximation, perfect for small datasets (<100k chunks)
#
# Architecture inside this class:
#
#   ┌─────────────────────────────────────┐
#   │  VectorStore                        │
#   │  ├── faiss_index  (FAISS object)    │  ← stores float32 vectors
#   │  └── chunks       (List[Chunk])     │  ← parallel list of metadata
#   │                                     │
#   │  add(chunks, vectors)               │
#   │  search(query_vec, top_k) → chunks  │
#   └─────────────────────────────────────┘
#
# The parallel list is crucial: FAISS returns integer indices (0, 1, 2, ...),
# and we look up the corresponding Chunk in self.chunks[index].
# =============================================================================

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from .chunker import Chunk
from .metadata_filter import MetadataFilter


class VectorStore:
    """
    In-memory FAISS-backed vector store.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the embedding vectors (e.g. 384 for MiniLM).
    """

    def __init__(self, embedding_dim: int = 384):
        import faiss  # imported here so the rest of the project loads without faiss
        self.embedding_dim = embedding_dim
        # IndexFlatIP = exact inner-product (cosine when normalised)
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._chunks: List[Chunk] = []  # parallel list to FAISS index rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """
        Add chunks and their vectors to the store.

        Parameters
        ----------
        chunks  : list of Chunk objects (metadata)
        vectors : numpy array of shape (N, embedding_dim)
        """
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks and vectors must match.")

        vectors = vectors.astype(np.float32)  # FAISS requires float32
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_spec: Optional[MetadataFilter] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Find the top-K most similar chunks to `query_vector`.

        Parameters
        ----------
        query_vector : numpy array of shape (embedding_dim,)
        top_k        : int – number of chunks to return
        filter_spec  : Optional[MetadataFilter] – metadata pre-filter rules

        Returns
        -------
        List of (Chunk, score) tuples, sorted by descending similarity.
        """
        if self._index.ntotal == 0:
            return []

        # If no filter or filter has no active rules, run fast FAISS index search directly
        if filter_spec is None or not filter_spec.has_rules():
            top_k_count = min(top_k, self._index.ntotal)
            query = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(query, top_k_count)

            results: List[Tuple[Chunk, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx == -1:
                    continue
                results.append((self._chunks[idx], float(score)))
            return results

        # Pre-filtering: Filter candidate indices using metadata matching
        candidate_indices = [
            i for i, chunk in enumerate(self._chunks)
            if filter_spec.matches(chunk.metadata)
        ]

        if not candidate_indices:
            return []

        # Reconstruct vectors for matching candidates and compute dot products
        # FAISS Flat index vector reconstruction:
        candidate_vectors = np.array([
            self._index.reconstruct(i) for i in candidate_indices
        ], dtype=np.float32)

        # Inner product (cosine similarity)
        query_vec = query_vector.astype(np.float32)
        scores = np.dot(candidate_vectors, query_vec)

        # Sort candidates descending by score
        sorted_pairs = sorted(
            zip(candidate_indices, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [(self._chunks[idx], float(score)) for idx, score in sorted_pairs]

    @property
    def total_chunks(self) -> int:
        """How many chunks are currently stored."""
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"VectorStore(dim={self.embedding_dim}, "
            f"chunks={self.total_chunks})"
        )
