# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# PURPOSE: Run the complete Agentic RAG pipeline end-to-end.
#
# What this script does step by step:
#   1. Load & chunk the documents from documents/
#   2. Embed the chunks using SentenceTransformers
#   3. Store embeddings in FAISS
#   4. Create the retriever, tools, memory, and LLM client
#   5. Run the Agentic RAG loop with a set of demo questions
#
# Run with:
#   python main.py
#
# First time run: SentenceTransformers downloads 'all-MiniLM-L6-v2' (~80 MB).
# Subsequent runs use the cached model.

# Suppress TensorFlow / oneDNN noise before any TF import happens
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # 0=all, 1=info, 2=warn, 3=error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # disable oneDNN messages
os.environ["ABSL_MIN_LOG_LEVEL"]    = "3"      # suppress absl logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Load .env file if present (picks up OPENAI_API_KEY)
load_dotenv()

# Add project root to path so imports work from any directory
sys.path.insert(0, str(Path(__file__).parent))

from core import Chunker, Embedder, VectorStore, Retriever, Evaluator, MetadataFilter
from agent import AgenticRAG, Memory, ToolRegistry
from agent.tools import make_search_tool, make_summarize_tool
from llm import LLMClient

console = Console()

# =============================================================================
# Configuration
# =============================================================================
DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks
TOP_K         = 3     # how many chunks to retrieve per query

# Demo questions paired with target topics for filtered search demo
DEMO_QUESTIONS = [
    ("What is Agentic RAG and how does it differ from basic RAG?", "agents"),
    ("Explain how FAISS is used in a vector database.", "ml"),
    ("What is the sliding window chunking strategy?", "ml"),
    ("How does the agent loop work? Describe each step.", "agents"),
    ("What is Retrieval-Augmented Generation (RAG)?", "basics"),
]


# =============================================================================
# Step 1: Ingestion — Load → Chunk → Embed → Store
# =============================================================================

def ingest_documents(chunker: Chunker, embedder: Embedder, store: VectorStore) -> int:
    """Load all .txt files from DOCUMENTS_DIR, chunk and embed them with metadata."""
    doc_files = list(DOCUMENTS_DIR.glob("*.txt"))

    if not doc_files:
        console.print(
            f"[bold red]No .txt files found in {DOCUMENTS_DIR}[/bold red]\n"
            "Please add documents to the documents/ directory."
        )
        return 0

    all_chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking documents...", total=len(doc_files))
        for doc_file in doc_files:
            chunks = chunker.chunk_file(str(doc_file))
            all_chunks.extend(chunks)
            progress.advance(task)
            topic = chunks[0].metadata.get("topic", "none") if chunks else "none"
            console.print(
                f"  [green]✓[/green] {doc_file.name}  "
                f"→  [cyan]{len(chunks)} chunks[/cyan]  "
                f"[dim](topic: [bold]{topic}[/bold])[/dim]"
            )

    console.print(
        f"\n[bold]Total chunks:[/bold] [cyan]{len(all_chunks)}[/cyan]  "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n"
    )

    # Embed all chunks in one batch
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            f"Embedding {len(all_chunks)} chunks with {embedder.model_name}..."
        )
        vectors = embedder.embed_chunks(all_chunks)

    console.print(
        f"  [green]✓[/green] Embeddings shape: [cyan]{vectors.shape}[/cyan]  "
        f"(float32, L2-normalised)\n"
    )

    # Add to FAISS index
    store.add(all_chunks, vectors)
    console.print(f"  [green]✓[/green] FAISS index now holds [cyan]{store.total_chunks}[/cyan] vectors\n")

    return len(all_chunks)


# =============================================================================
# Step 2: Build the agent
# =============================================================================

def build_agent(retriever: Retriever, llm: LLMClient) -> AgenticRAG:
    """Wire up tools, memory, and the agent."""
    registry = ToolRegistry()
    registry.register(make_search_tool(retriever))
    registry.register(make_summarize_tool())

    memory = Memory(max_history=20)

    return AgenticRAG(
        llm=llm,
        tool_registry=registry,
        memory=memory,
        max_iterations=5,
        verbose=True,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    console.clear()
    console.print(Rule("[bold bright_blue]🚀  Agentic RAG — Metadata Filtering & Evaluation Demo[/bold bright_blue]"))
    console.print()

    # ── Initialise components ──────────────────────────────────────────
    console.print(Rule("[yellow]⚙️   Step 1: Initialising components[/yellow]"))
    chunker   = Chunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    embedder  = Embedder()            # uses 'all-MiniLM-L6-v2'
    store     = VectorStore(embedding_dim=384)
    retriever = Retriever(embedder, store, top_k=TOP_K, min_score=0.0)
    evaluator = Evaluator(embedder)
    llm       = LLMClient()

    console.print(
        f"  Chunker    : chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n"
        f"  Embedder   : {embedder.model_name}\n"
        f"  VectorStore: FAISS IndexFlatIP + Metadata Pre-Filtering\n"
        f"  Evaluator  : 4 Metrics (Context Rel, Faithfulness, Answer Rel, Context Recall)\n"
        f"  LLM        : {llm}\n"
    )

    # ── Ingest documents ───────────────────────────────────────────────
    console.print(Rule("[yellow]📄  Step 2: Loading & Indexing Documents with Metadata[/yellow]"))
    n_chunks = ingest_documents(chunker, embedder, store)

    if n_chunks == 0:
        console.print("[red]Ingestion failed. Exiting.[/red]")
        return

    # ── Metadata Filtering Demo ────────────────────────────────────────
    console.print(Rule("[yellow]🔍  Step 3: Metadata Pre-Filtering Demo[/yellow]"))
    sample_query = "What is the agent loop?"
    
    unfiltered = retriever.retrieve(sample_query, top_k=2)
    filtered = retriever.retrieve(sample_query, top_k=2, filter_spec=MetadataFilter(topic="agents"))

    console.print(f"  Query: [bold white]'{sample_query}'[/bold white]")
    console.print(f"  Unfiltered Search  → [cyan]{len(unfiltered)} chunks retrieved[/cyan] (all topics)")
    console.print(f"  Filtered Search    → [cyan]{len(filtered)} chunks retrieved[/cyan] (topic='agents')")
    for r in filtered:
        console.print(f"    • [green]Match[/green]: {r.chunk.source} | topic={r.chunk.metadata.get('topic')}")
    console.print()

    # ── Build agent ────────────────────────────────────────────────────
    console.print(Rule("[yellow]🤖  Step 4: Building the Agentic RAG Agent[/yellow]"))
    agent = build_agent(retriever, llm)
    console.print(
        f"  [green]✓[/green] Agent ready  "
        f"(tools: {agent.tools.list_tools()}, "
        f"max_iter={agent.max_iterations})\n"
    )

    # ── Run demo questions & evaluation ───────────────────────────────
    console.print(Rule("[yellow]💬  Step 5: Running Agentic Loop + Live Evaluation[/yellow]"))
    console.print(
        f"  Running [bold]{len(DEMO_QUESTIONS)}[/bold] demo questions.\n"
        f"  LLM mode: [bold]{'🟡 MOCK (offline)' if llm.is_mock else '🟢 OpenAI'}[/bold]\n"
    )

    for i, (question, target_topic) in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"\n[bold dim]{'─'*60}[/bold dim]")
        console.print(f"[bold dim]Question {i}/{len(DEMO_QUESTIONS)}[/bold dim]")
        
        # Reset memory between questions
        agent.memory.clear_all()
        answer = agent.ask(question)
        console.print()
        console.print(
            Panel(
                f"[bold green]{answer}[/bold green]",
                title=f"[bold green]✅ Final Answer ({i}/{len(DEMO_QUESTIONS)})[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        # Retrieve context for evaluation
        filter_spec = MetadataFilter(topic=target_topic)
        retrieved = retriever.retrieve(question, top_k=3, filter_spec=filter_spec)

        # Run Live Evaluator
        eval_res = evaluator.evaluate(query=question, answer=answer, retrieval_results=retrieved)
        evaluator.print_eval_report(eval_res)

    # ── Summary ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold bright_blue]✅  Pipeline Complete![/bold bright_blue]"))
    console.print(
        f"\n  [dim]Documents indexed : {n_chunks} chunks[/dim]\n"
        f"  [dim]Questions answered: {len(DEMO_QUESTIONS)}[/dim]\n"
        f"  [dim]Run standalone evaluation: [bold]python eval_runner.py[/bold][/dim]\n"
        f"  [dim]View project structure: [bold]python show_structure.py[/bold][/dim]\n"
    )


if __name__ == "__main__":
    main()
