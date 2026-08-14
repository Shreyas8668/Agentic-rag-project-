# =============================================================================
# core/retriever.py
# -----------------------------------------------------------------------------
# CONCEPT: Retrieval
# The retriever is the bridge between a user query and the knowledge base.
# It orchestrates:
#   1. Embed the query → query_vector
#   2. Search the vector store → top-K (chunk, score) pairs
#   3. (Optional) filter by minimum score threshold
#   4. Return formatted results for the agent to read
#
# This is the component the agent calls when it uses the "search" tool.
#
# RAG quality heavily depends on retrieval quality — if you retrieve the wrong
# chunks, even a perfect LLM will give a wrong answer ("garbage in, garbage out").
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from .chunker import Chunk
from .embedder import Embedder
from .vector_store import VectorStore
from .metadata_filter import MetadataFilter


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float         # cosine similarity in [-1, 1]
    rank: int            # 1 = most similar

    def __str__(self) -> str:
        topic_tag = f" | Topic: {self.chunk.metadata.get('topic')}" if self.chunk.metadata.get('topic') else ""
        return (
            f"[Rank {self.rank} | Score {self.score:.3f} | "
            f"Source: {self.chunk.source}{topic_tag}]\n{self.chunk.text}"
        )


class Retriever:
    """
    Semantic retriever: converts a query to a vector and fetches top-K chunks.

    Parameters
    ----------
    embedder      : Embedder – produces query vectors
    vector_store  : VectorStore – holds chunk vectors
    top_k         : int – default number of results to return
    min_score     : float – discard results below this similarity (0.0 = keep all)
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.0,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_spec: Optional[MetadataFilter] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks for `query`, with optional metadata pre-filtering.

        Parameters
        ----------
        query       : str – the user's question or search phrase
        top_k       : int – override the default top_k for this call
        filter_spec : Optional[MetadataFilter] – metadata pre-filtering rules

        Returns
        -------
        List[RetrievalResult] sorted by descending score (best match first).
        """
        k = top_k or self.top_k

        # Step 1: Embed the query into a vector
        query_vector = self.embedder.embed_query(query)

        # Step 2: Search the vector store with optional metadata filter
        raw_results = self.vector_store.search(
            query_vector, top_k=k, filter_spec=filter_spec
        )

        # Step 3: Filter by minimum score and wrap in RetrievalResult
        results: List[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(raw_results, start=1):
            if score >= self.min_score:
                results.append(
                    RetrievalResult(chunk=chunk, score=score, rank=rank)
                )

        return results

    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Concatenate retrieved chunks into a single context string
        ready to be injected into an LLM prompt.
        """
        if not results:
            return "No relevant information found in the knowledge base."

        sections = []
        for r in results:
            sections.append(
                f"--- Source: {r.chunk.source} (chunk {r.chunk.chunk_id}) ---\n"
                f"{r.chunk.text}"
            )
        return "\n\n".join(sections)
