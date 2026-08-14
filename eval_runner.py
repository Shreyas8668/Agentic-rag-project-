# =============================================================================
# eval_runner.py
# -----------------------------------------------------------------------------
# PURPOSE: Dedicated RAG Evaluation Runner
# Evaluates the RAG pipeline across test cases and produces comprehensive
# quality metrics and visual terminal scorecards.
#
# Run with:
#   python eval_runner.py
# =============================================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows terminal encoding for Rich output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from core import Chunker, Embedder, VectorStore, Retriever, Evaluator, MetadataFilter
from agent import AgenticRAG, Memory, ToolRegistry
from agent.tools import make_search_tool, make_summarize_tool
from llm import LLMClient

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console(file=sys.stdout, force_terminal=True)

# -----------------------------------------------------------------------------
# Test Cases for Evaluation (Query, Reference Topic, Optional Ground Truth)
# -----------------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "What is Retrieval-Augmented Generation (RAG)?",
        "topic_filter": "basics",
        "ground_truth": "RAG combines a retrieval system with an LLM to search a knowledge base before generating answers.",
    },
    {
        "query": "How does the Agentic Think-Act-Observe loop work?",
        "topic_filter": "agents",
        "ground_truth": "The agent reads the question, calls tools like search_knowledge_base, observes results, and iterates until confident.",
    },
    {
        "query": "What is FAISS and why do we use vector embeddings?",
        "topic_filter": "ml",
        "ground_truth": "Vector embeddings represent semantic text as float lists. FAISS searches millions of vectors in milliseconds.",
    },
    {
        "query": "What strategies are used for splitting long documents into chunks?",
        "topic_filter": "ml",
        "ground_truth": "Chunking strategies include fixed-size sliding windows with overlap, sentence-based, and recursive character splits.",
    },
]


def main():
    console.clear()
    console.print(Rule("[bold bright_blue]📊 RAG Evaluation Suite — Benchmark & Quality Audit[/bold bright_blue]"))
    console.print()

    # 1. Initialize RAG components
    console.print("[dim]Initialising embedding model and vector store...[/dim]")
    chunker = Chunker(chunk_size=500, overlap=100)
    embedder = Embedder()
    store = VectorStore(embedding_dim=384)
    retriever = Retriever(embedder, store, top_k=3)
    evaluator = Evaluator(embedder)
    llm = LLMClient()

    # 2. Ingest documents
    doc_dir = Path(__file__).parent / "documents"
    all_chunks = []
    for txt_file in doc_dir.glob("*.txt"):
        all_chunks.extend(chunker.chunk_file(str(txt_file)))

    vectors = embedder.embed_chunks(all_chunks)
    store.add(all_chunks, vectors)
    console.print(f"[green]✓[/green] Ingested [cyan]{len(all_chunks)} chunks[/cyan] across [cyan]{len(list(doc_dir.glob('*.txt')))} files[/cyan]\n")

    # 3. Build Agent
    registry = ToolRegistry()
    registry.register(make_search_tool(retriever))
    registry.register(make_summarize_tool())
    agent = AgenticRAG(llm=llm, tool_registry=registry, verbose=False)

    # 4. Run Evaluation Benchmark
    results = []

    for i, test in enumerate(TEST_CASES, start=1):
        console.print(Rule(f"[yellow]Test Case {i}/{len(TEST_CASES)}: {test['query']}[/yellow]"))
        
        # Reset memory for isolated testing
        agent.memory.clear_all()

        # Step A: Perform retrieval (with optional topic filter)
        filter_spec = MetadataFilter(topic=test["topic_filter"]) if test.get("topic_filter") else None
        retrieved = retriever.retrieve(test["query"], top_k=3, filter_spec=filter_spec)

        # Step B: Get Agent Answer
        answer = agent.ask(test["query"])

        # Step C: Compute Evaluation Metrics
        eval_res = evaluator.evaluate(
            query=test["query"],
            answer=answer,
            retrieval_results=retrieved,
            ground_truth=test.get("ground_truth"),
        )
        results.append(eval_res)

        # Step D: Print detailed metric card
        evaluator.print_eval_report(eval_res)
        console.print()

    # 5. Print Summary Benchmarking Table
    summary_table = Table(
        title="[bold bright_white]🏆 RAG Benchmark Summary Table[/bold bright_white]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        padding=(0, 2),
    )
    summary_table.add_column("#", justify="center")
    summary_table.add_column("Query Snippet", style="white", width=35)
    summary_table.add_column("Context Rel", justify="right")
    summary_table.add_column("Faithfulness", justify="right")
    summary_table.add_column("Answer Rel", justify="right")
    summary_table.add_column("Context Recall", justify="right")
    summary_table.add_column("Overall", justify="right", style="bold yellow")

    for idx, r in enumerate(results, start=1):
        summary_table.add_row(
            str(idx),
            r.query[:32] + "...",
            f"{r.context_relevance:.2f}",
            f"{r.faithfulness:.2f}",
            f"{r.answer_relevance:.2f}",
            f"{r.context_recall:.2f}",
            f"{r.overall_score:.2f}",
        )

    # Compute averages
    avg_c_rel = sum(r.context_relevance for r in results) / len(results)
    avg_faith = sum(r.faithfulness for r in results) / len(results)
    avg_a_rel = sum(r.answer_relevance for r in results) / len(results)
    avg_c_rec = sum(r.context_recall for r in results) / len(results)
    avg_all = sum(r.overall_score for r in results) / len(results)

    summary_table.add_section()
    summary_table.add_row(
        "AVG",
        "[bold cyan]AVERAGE METRICS[/bold cyan]",
        f"[bold]{avg_c_rel:.2f}[/bold]",
        f"[bold]{avg_faith:.2f}[/bold]",
        f"[bold]{avg_a_rel:.2f}[/bold]",
        f"[bold]{avg_c_rec:.2f}[/bold]",
        f"[bold bright_green]{avg_all:.2f}[/bold bright_green]",
    )

    console.print(summary_table)
    console.print()
    console.print(Rule("[bold bright_blue]Evaluation Complete! 🚀[/bold bright_blue]"))


if __name__ == "__main__":
    main()
