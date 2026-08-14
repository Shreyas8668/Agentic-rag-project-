# =============================================================================
# show_structure.py
# -----------------------------------------------------------------------------
# PURPOSE: Visual learning aid — shows the ENTIRE project structure in the
#          terminal with colors, descriptions, and concept annotations.
#
# Run with:
#   python show_structure.py
#
# This uses the 'rich' library to produce a beautiful tree view that explains
# what each file does and which RAG concept it teaches.
# =============================================================================

# Fix Windows terminal encoding so Rich can render unicode characters
import sys, io, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich import box
from rich.rule import Rule


console = Console(file=sys.stdout, force_terminal=True, highlight=False)

# =============================================================================
# File descriptions: filename → (emoji, short role, concept taught)
# =============================================================================
FILE_META = {
    # Root files
    "app.py": (
        "🌐", "Web Dashboard", "Interactive Streamlit UI with thought inspector & charts"
    ),
    "main.py": (
        "🚀", "Entry point", "Orchestrates the full RAG pipeline"
    ),
    "eval_runner.py": (
        "📊", "Eval Runner", "Runs RAG evaluation benchmarks & scorecards"
    ),
    "show_structure.py": (
        "🗺️ ", "This file!", "Visual guide to the entire project"
    ),
    "requirements.txt": (
        "📦", "Dependencies", "All Python packages needed"
    ),
    ".env.example": (
        "🔑", "Config template", "Copy to .env and add your API key"
    ),

    # documents/
    "sample.txt": (
        "📄", "Knowledge base", "Raw text the agent learns from"
    ),
    "rag_basics.txt": (
        "📄", "Topic: basics", "RAG definition, stages, grounding"
    ),
    "agent_concepts.txt": (
        "📄", "Topic: agents", "Agentic loop, tools, memory"
    ),
    "ml_concepts.txt": (
        "📄", "Topic: ml", "Embeddings, FAISS, chunking"
    ),

    # core/
    "core/__init__.py": (
        "📂", "Package init", "Exports Chunker, Embedder, VectorStore, Retriever, Filter, Evaluator"
    ),
    "core/chunker.py": (
        "✂️ ", "Chunker", "Splits documents into overlapping text windows"
    ),
    "core/embedder.py": (
        "🧠", "Embedder", "Converts text → semantic float vectors (MiniLM)"
    ),
    "core/vector_store.py": (
        "🗄️ ", "Vector Store", "FAISS index: stores & pre-filters vectors"
    ),
    "core/retriever.py": (
        "🔍", "Retriever", "Embed query → pre-filtered top-K search"
    ),
    "core/metadata_filter.py": (
        "🏷️ ", "Metadata Filter", "Pre-filters vector candidate space by topic/date/source"
    ),
    "core/evaluator.py": (
        "📐", "RAG Evaluator", "4 metrics: Context Rel, Faithfulness, Answer Rel, Context Recall"
    ),

    # agent/
    "agent/__init__.py": (
        "📂", "Package init", "Exports AgenticRAG, Memory, ToolRegistry"
    ),
    "agent/agent.py": (
        "🤖", "Agent Loop", "Think → Act → Observe → Answer iteration"
    ),
    "agent/tools.py": (
        "🔧", "Tools", "search_knowledge_base + summarize tool definitions"
    ),
    "agent/memory.py": (
        "🧩", "Memory", "Conversation history + scratchpad working notes"
    ),

    # llm/
    "llm/__init__.py": (
        "📂", "Package init", "Exports LLMClient"
    ),
    "llm/llm_client.py": (
        "💬", "LLM Client", "OpenAI API wrapper with offline mock fallback"
    ),
}

# =============================================================================
# Concept → color mapping (for the concept pills)
# =============================================================================
CONCEPT_COLORS = {
    "Chunking":      "bright_cyan",
    "Embedding":     "bright_blue",
    "Vector Search": "bright_magenta",
    "Retrieval":     "bright_yellow",
    "Filtering":     "magenta",
    "Evaluation":    "bright_green",
    "Agent Loop":    "bright_red",
    "Tools":         "bright_green",
    "Memory":        "orange1",
    "LLM":           "bright_white",
    "Config":        "grey70",
    "Meta":          "grey50",
}

FILE_CONCEPTS = {
    "core/chunker.py":         "Chunking",
    "core/embedder.py":        "Embedding",
    "core/vector_store.py":    "Vector Search",
    "core/retriever.py":       "Retrieval",
    "core/metadata_filter.py": "Filtering",
    "core/evaluator.py":       "Evaluation",
    "eval_runner.py":          "Evaluation",
    "agent/agent.py":          "Agent Loop",
    "agent/tools.py":          "Tools",
    "agent/memory.py":         "Memory",
    "llm/llm_client.py":       "LLM",
    ".env.example":            "Config",
    "requirements.txt":        "Config",
}


def make_label(rel_path: str, name: str) -> Text:
    """Build a rich Text object for a file node."""
    meta = FILE_META.get(rel_path) or FILE_META.get(name)
    if meta:
        emoji, role, description = meta
        t = Text()
        t.append(f"{emoji}  ", style="bold")
        t.append(f"{name}", style="bold white")
        t.append(f"  —  ", style="dim")
        t.append(role, style="italic cyan")
        t.append(f"  ·  {description}", style="dim white")

        concept = FILE_CONCEPTS.get(rel_path)
        if concept:
            color = CONCEPT_COLORS.get(concept, "white")
            t.append(f"  [{concept}]", style=f"bold {color}")
        return t
    else:
        return Text(f"📁  {name}", style="bold white")


def build_tree() -> Tree:
    """Walk the project directory and build a Rich Tree."""
    root_path = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(root_path)

    root_label = Text()
    root_label.append("🗂️  ", style="bold")
    root_label.append(project_name, style="bold bright_white")
    root_label.append("  —  Agentic RAG (from scratch)", style="dim white")

    tree = Tree(root_label, guide_style="bold bright_blue")

    # Define the order of directories we want to show
    ordered_dirs = ["documents", "core", "agent", "llm"]
    root_files = ["app.py", "main.py", "eval_runner.py", "show_structure.py", "requirements.txt", ".env.example"]

    dir_icons = {
        "documents": "📁",
        "core":      "⚙️ ",
        "agent":     "🤖",
        "llm":       "💬",
    }
    dir_descriptions = {
        "documents": "Raw knowledge base files with topic tags",
        "core":      "Chunking · Embedding · Vector Store · Pre-Filtering · Evaluator",
        "agent":     "Agent loop · Tools · Memory",
        "llm":       "LLM abstraction layer",
    }

    # -- Root-level files --
    for fname in root_files:
        fpath = os.path.join(root_path, fname)
        exists = os.path.isfile(fpath)
        label = make_label(fname, fname)
        if not exists:
            label.append(" ⚠ (not created yet)", style="bold red")
        tree.add(label)

    # -- Sub-directories --
    for dname in ordered_dirs:
        dpath = os.path.join(root_path, dname)
        icon = dir_icons.get(dname, "📁")
        desc = dir_descriptions.get(dname, "")

        dir_label = Text()
        dir_label.append(f"{icon}  ", style="bold")
        dir_label.append(f"{dname}/", style="bold yellow")
        dir_label.append(f"  —  {desc}", style="dim white")

        branch = tree.add(dir_label)

        if os.path.isdir(dpath):
            for fname in sorted(os.listdir(dpath)):
                if fname.startswith("__pycache__") or fname.endswith(".pyc"):
                    continue
                rel = f"{dname}/{fname}"
                label = make_label(rel, fname)
                branch.add(label)
        else:
            branch.add(Text("  (directory not found)", style="red"))

    return tree


def print_concept_legend() -> None:
    """Print a color-coded legend of RAG concepts."""
    table = Table(
        title="[bold white]Agentic RAG — Concept Map[/bold white]",
        box=box.ROUNDED,
        border_style="bright_blue",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
    )
    table.add_column("Step #", style="bold white", justify="center")
    table.add_column("Concept", style="bold")
    table.add_column("File", style="cyan")
    table.add_column("What it does", style="dim white")

    rows = [
        ("1", "Chunking",      "core/chunker.py",         "Split raw text → overlapping windows + tags"),
        ("2", "Embedding",     "core/embedder.py",        "Text → 384-dim float vector (MiniLM)"),
        ("3", "Vector Search", "core/vector_store.py",    "FAISS index: store & pre-filter vectors"),
        ("4", "Filtering",     "core/metadata_filter.py", "Pre-filter candidates by topic/date/source"),
        ("5", "Retrieval",     "core/retriever.py",       "query → pre-filtered top-K cosine search"),
        ("6", "Evaluation",    "core/evaluator.py",       "Context Rel, Faithfulness, Answer Rel, Context Recall"),
        ("7", "Memory",        "agent/memory.py",         "Chat history + scratchpad notes"),
        ("8", "Tools",         "agent/tools.py",          "search_knowledge_base (with topic filter), summarize"),
        ("9", "Agent Loop",    "agent/agent.py",          "Think → Act → Observe → Answer"),
        ("10", "LLM",          "llm/llm_client.py",       "OpenAI / offline mock LLM calls"),
    ]

    for step, concept, file_, what in rows:
        color = CONCEPT_COLORS.get(concept, "white")
        table.add_row(
            step,
            Text(concept, style=f"bold {color}"),
            file_,
            what,
        )

    console.print(table)


def print_pipeline_diagram() -> None:
    """Print an ASCII art pipeline of the RAG flow."""
    diagram = """
  [bold bright_white]DOCUMENTS[/bold bright_white]
       │
       ▼
  [bright_cyan]CHUNKER[/bright_cyan]   ─── Split into overlapping windows + metadata tags
       │
       ▼
  [bright_blue]EMBEDDER[/bright_blue]  ─── Text → float vectors (MiniLM)
       │
       ▼
  [bright_magenta]VECTOR STORE[/bright_magenta] ─ FAISS index + [magenta]Metadata Pre-Filtering[/magenta]
       │
       │ (at query time)
       ▼
  [bright_yellow]RETRIEVER[/bright_yellow] ─── query + FilterSpec → top-K search
       │
       ▼
  [bold bright_white]AGENT LOOP[/bold bright_white] ─── Think → Act → Observe → Answer
       │
       ▼
  [bright_green]EVALUATOR[/bright_green]  ─── Context Rel · Faithfulness · Answer Rel · Recall
"""
    panel = Panel(
        diagram,
        title="[bold blue]Agentic RAG — Pipeline Flow with Filtering & Evaluation[/bold blue]",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 4),
    )
    console.print(panel)


def main():
    console.clear()
    console.print(Rule("[bold bright_blue]🧠  Agentic RAG — Project Structure Viewer[/bold bright_blue]"))
    console.print()

    # 1. Pipeline diagram
    print_pipeline_diagram()
    console.print()

    # 2. File tree
    console.print(Rule("[bold yellow]📂  Project File Tree[/bold yellow]"))
    console.print()
    tree = build_tree()
    console.print(tree)
    console.print()

    # 3. Concept legend
    console.print(Rule("[bold cyan]📚  Concept Map[/bold cyan]"))
    console.print()
    print_concept_legend()
    console.print()

    # 4. Quick-start commands
    qs_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    qs_table.add_column("Command", style="bold green")
    qs_table.add_column("What it does", style="dim white")
    qs_table.add_row("streamlit run app.py",      "Launch Interactive Web Dashboard UI")
    qs_table.add_row("python show_structure.py", "Show this visualization")
    qs_table.add_row("python main.py",            "Run full Agentic RAG pipeline + live eval")
    qs_table.add_row("python eval_runner.py",     "Run dedicated RAG evaluation benchmark")
    qs_table.add_row("pip install -r requirements.txt", "Install dependencies")
    qs_table.add_row("copy .env.example .env",    "Set up your OpenAI API key")

    console.print(Panel(
        qs_table,
        title="[bold green]⚡  Quick Start Commands[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    ))
    console.print()
    console.print(Rule("[bold bright_blue]Happy Learning! 🚀[/bold bright_blue]"))


if __name__ == "__main__":
    main()
