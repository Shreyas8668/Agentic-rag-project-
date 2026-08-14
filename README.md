# 🧠 Agentic RAG — From Scratch with Metadata Filtering, Evaluation & Web UI

A complete, educational **Agentic Retrieval-Augmented Generation (RAG)** system built from scratch in Python. Demonstrates end-to-end RAG concepts: sliding-window chunking, vector embeddings, FAISS indexing, metadata pre-filtering, an autonomous **Think → Act → Observe → Answer** agent reasoning loop, a 4-metric evaluation engine, and an interactive Streamlit web dashboard.

---

## 🌟 Key Features

- **⚡ Core RAG Pipeline**: Sliding window chunker with overlap, SentenceTransformer embeddings (`all-MiniLM-L6-v2`), and FAISS flat vector index.
- **🏷️ Metadata Pre-Filtering**: Restrict vector search space by topic (`basics`, `agents`, `ml`), source file, or date range before vector similarity ranking.
- **🤖 Autonomous Agentic Loop**: Think → Act → Observe reasoning cycle with tool calling (`search_knowledge_base`, `summarize`) and conversation memory.
- **📐 4-Metric RAG Evaluator**: 100% local evaluation engine measuring **Context Relevance**, **Faithfulness**, **Answer Relevance**, and **Context Recall** without external LLM judges.
- **🌐 Interactive Streamlit UI (`app.py`)**: Live agent thought-inspector, metadata filter controls, chunk inspector, and metric scorecards.
- **🗺️ Terminal Visualizer (`show_structure.py`)**: Rich tree view and pipeline flow mapping every concept to code.

---

## 🗂️ Project Structure

```text
agentic_rag/
├── app.py               ← Interactive Streamlit Web UI (Dashboard)
├── main.py              ← Pipeline demo & ingestion entry point
├── eval_runner.py       ← Dedicated RAG evaluation benchmark suite
├── show_structure.py    ← Terminal visualizer (Rich file tree & concept map)
├── requirements.txt     ← Project dependencies
├── .env.example         ← Environment configuration template
│
├── documents/           ← Knowledge base text files with metadata tags
│   ├── rag_basics.txt   (Topic: basics)
│   ├── agent_concepts.txt (Topic: agents)
│   ├── ml_concepts.txt  (Topic: ml)
│   └── sample.txt
│
├── core/                ← RAG Engine Modules
│   ├── chunker.py       ← Sliding window chunker with metadata tag extraction
│   ├── embedder.py      ← MiniLM embedding wrapper (384-dimensional)
│   ├── vector_store.py  ← FAISS flat vector index + metadata pre-filtering
│   ├── retriever.py     ← Cosine similarity search with MetadataFilter
│   ├── metadata_filter.py ← MetadataFilter dataclass & matching logic
│   └── evaluator.py     ← 4-Metric RAG Evaluator & Rich scorecards
│
├── agent/               ← Agentic Reasoning Brain
│   ├── agent.py         ← Autonomous Think → Act → Observe loop
│   ├── tools.py         ← Tool registry (search_knowledge_base with topic filtering)
│   └── memory.py        ← Short-term conversation memory & scratchpad
│
└── llm/                 ← LLM Abstraction Layer
    └── llm_client.py    ← OpenAI API wrapper + offline mock LLM fallback
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Shreyas8668/Agentic-rag-project-.git
cd Agentic-rag-project-
pip install -r requirements.txt
```

### 2. Configuration (Optional for real LLM)
```bash
copy .env.example .env
# Edit .env and paste your OPENAI_API_KEY
```
*Note: If no API key is provided, the project runs in offline mock LLM mode.*

---

## 🖥️ Usage & Commands

| Command | Description |
|---|---|
| `streamlit run app.py` | Launch interactive web UI at `http://localhost:8501` |
| `python show_structure.py` | Display colorful project structure & concept map in terminal |
| `python main.py` | Run end-to-end pipeline demo with live evaluation |
| `python eval_runner.py` | Execute automated RAG evaluation benchmark test cases |

---

## 📚 Concepts Taught

1. **Chunking**: Preserving context at boundaries using overlapping sliding windows.
2. **Embeddings**: Mapping text to 384-dimensional dense semantic float vectors.
3. **Vector Search**: Efficient similarity retrieval using FAISS exact inner-product search.
4. **Metadata Filtering**: Narrowing vector candidate space before similarity calculation.
5. **Agentic Reasoning**: Loop enabling LLMs to decide when, what, and how to search before answering.
6. **RAG Evaluation**: Quantitative assessment of grounding, relevancy, and context recall.
