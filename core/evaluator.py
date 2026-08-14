# =============================================================================
# core/evaluator.py
# -----------------------------------------------------------------------------
# CONCEPT: RAG Evaluation
# Evaluating RAG applications is essential to ensure quality and prevent hallucination.
#
# We implement 4 core RAG metrics (aligned with industry frameworks like RAGAS):
#
#   1. Context Relevance  – Are retrieved chunks relevant to the query?
#   2. Faithfulness       – Is the generated answer grounded ONLY in the retrieved context?
#   3. Answer Relevance   – Does the answer directly address the user's question?
#   4. Context Recall     – Did retrieval fetch enough relevant context to answer fully?
#
# All metrics use local embeddings (cosine similarity) and token overlap calculations,
# requiring NO external LLM judge or extra API calls.
# =============================================================================

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

from .embedder import Embedder
from .retriever import RetrievalResult

console = Console()


@dataclass
class EvalResult:
    """Evaluation scores for a single RAG query-response turn."""
    query: str
    answer: str
    retrieved_chunks: List[str]
    
    # 4 Core Metrics (scores normalized 0.0 to 1.0)
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    context_recall: float

    @property
    def overall_score(self) -> float:
        """Weighted aggregate score across all 4 metrics."""
        return float(np.mean([
            self.context_relevance,
            self.faithfulness,
            self.answer_relevance,
            self.context_recall,
        ]))

    def to_dict(self) -> Dict[str, float]:
        return {
            "context_relevance": round(self.context_relevance, 3),
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_recall": round(self.context_recall, 3),
            "overall_score": round(self.overall_score, 3),
        }


class Evaluator:
    """
    Local RAG Evaluator using Embedder vector similarity and token overlap analysis.

    Parameters
    ----------
    embedder : Embedder
        The project's SentenceTransformer embedder instance.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieval_results: List[RetrievalResult],
        ground_truth: Optional[str] = None,
    ) -> EvalResult:
        """
        Evaluate a complete RAG interaction.

        Parameters
        ----------
        query : str
            The original user question.
        answer : str
            The agent's final generated answer.
        retrieval_results : List[RetrievalResult]
            The list of retrieved chunk objects.
        ground_truth : Optional[str]
            Optional reference answer for deeper recall checking.

        Returns
        -------
        EvalResult object containing all 4 metric scores.
        """
        context_texts = [r.chunk.text for r in retrieval_results]
        full_context = "\n\n".join(context_texts)

        # Calculate individual metric scores
        c_rel = self._calc_context_relevance(query, context_texts)
        faith = self._calc_faithfulness(answer, full_context)
        a_rel = self._calc_answer_relevance(query, answer)
        c_rec = self._calc_context_recall(query, context_texts, ground_truth)

        return EvalResult(
            query=query,
            answer=answer,
            retrieved_chunks=context_texts,
            context_relevance=c_rel,
            faithfulness=faith,
            answer_relevance=a_rel,
            context_recall=c_rec,
        )

    def print_eval_report(self, result: EvalResult) -> None:
        """Display a pretty Rich terminal panel showing evaluation metrics."""
        metrics = [
            ("Context Relevance", result.context_relevance, "Relevance of retrieved chunks to query"),
            ("Faithfulness",      result.faithfulness,      "Grounding of answer in retrieved context"),
            ("Answer Relevance",  result.answer_relevance,  "Directness of answer to query"),
            ("Context Recall",     result.context_recall,     "Coverage of query topics in retrieved text"),
        ]

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 2),
        )
        table.add_column("Metric", style="bold white", width=20)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Visual", width=20)
        table.add_column("Description", style="dim white")

        for name, score, desc in metrics:
            bar_len = int(score * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            
            # Color code score
            if score >= 0.75:
                color = "green"
            elif score >= 0.5:
                color = "yellow"
            else:
                color = "red"

            table.add_row(
                name,
                f"[{color}]{score:.2f}[/{color}]",
                f"[{color}]{bar}[/{color}]",
                desc,
            )

        overall = result.overall_score
        stars = "★" * int(overall * 5) + "☆" * (5 - int(overall * 5))
        overall_color = "green" if overall >= 0.7 else ("yellow" if overall >= 0.5 else "red")

        report_panel = Panel(
            table,
            title=f"[bold blue]📊 RAG Evaluation Report | Overall: [{overall_color}]{overall:.2f} {stars}[/{overall_color}][/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(report_panel)

    # ------------------------------------------------------------------
    # Metric Calculation Details
    # ------------------------------------------------------------------

    def _calc_context_relevance(self, query: str, chunks: List[str]) -> float:
        """
        Context Relevance = average cosine similarity between query embedding
        and each retrieved chunk embedding.
        """
        if not chunks:
            return 0.0

        q_vec = self.embedder.embed_query(query)
        sims = []
        for text in chunks:
            chunk_vec = self.embedder.embed_query(text)
            # Dot product of normalized vectors = cosine similarity
            sim = float(np.dot(q_vec, chunk_vec))
            sims.append(max(0.0, sim))  # clamp negative scores to 0

        return float(np.mean(sims)) if sims else 0.0

    def _calc_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness = proportion of key statements/nouns/verbs in the answer
        that can be found in the retrieved context (token/n-gram overlap).
        """
        if not answer or not context:
            return 0.0

        # Tokenize answer into meaningful keywords (strip stop words & punctuation)
        ans_tokens = self._extract_keywords(answer)
        if not ans_tokens:
            return 1.0  # Empty or generic answer

        ctx_text_lower = context.lower()
        matched = sum(1 for token in ans_tokens if token in ctx_text_lower)

        return float(matched / len(ans_tokens))

    def _calc_answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance = cosine similarity between query vector and answer vector.
        """
        if not query or not answer:
            return 0.0

        # Strip prefixes like "FINAL ANSWER:" for clean embedding
        clean_ans = answer.replace("FINAL ANSWER:", "").strip()

        q_vec = self.embedder.embed_query(query)
        ans_vec = self.embedder.embed_query(clean_ans)

        sim = float(np.dot(q_vec, ans_vec))
        return max(0.0, float(sim))

    def _calc_context_recall(
        self, query: str, chunks: List[str], ground_truth: Optional[str] = None
    ) -> float:
        """
        Context Recall = how well the retrieved chunks cover the query's key terms
        (or ground truth statement if provided).
        """
        target_text = ground_truth if ground_truth else query
        query_keywords = self._extract_keywords(target_text)

        if not query_keywords:
            return 1.0

        full_context = " ".join(chunks).lower()
        found = sum(1 for kw in query_keywords if kw in full_context)

        return float(found / len(query_keywords))

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract normalized words excluding common English stop words."""
        STOP_WORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "to", "from", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now", "it", "its", "this",
            "that", "these", "those", "what", "which", "who", "whom", "of", "and"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in STOP_WORDS]
