# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# PURPOSE: Interactive Streamlit Web Application for Agentic RAG
# Includes live agent thought-inspector, metadata filtering controls,
# chunk inspector, and 4-metric RAG evaluation scorecards.
#
# Run with:
#   streamlit run app.py
# =============================================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core import Chunker, Embedder, VectorStore, Retriever, Evaluator, MetadataFilter
from agent import AgenticRAG, Memory, ToolRegistry
from agent.tools import make_search_tool, make_summarize_tool
from llm import LLMClient

# -----------------------------------------------------------------------------
# Streamlit Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG — Interactive Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: #1E232F;
        border: 1px solid #2E364A;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .chunk-box {
        background: #161B26;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .badge-topic {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-score {
        background-color: #065F46;
        color: #34D399;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached RAG Engine Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def init_rag_system():
    """Load and index documents into FAISS vector store once per app lifecycle."""
    chunker = Chunker(chunk_size=500, overlap=100)
    embedder = Embedder()
    store = VectorStore(embedding_dim=384)
    evaluator = Evaluator(embedder)
    llm = LLMClient()

    doc_dir = Path(__file__).parent / "documents"
    all_chunks = []

    for txt_file in doc_dir.glob("*.txt"):
        chunks = chunker.chunk_file(str(txt_file))
        all_chunks.extend(chunks)

    if all_chunks:
        vectors = embedder.embed_chunks(all_chunks)
        store.add(all_chunks, vectors)

    retriever = Retriever(embedder, store, top_k=3, min_score=0.0)

    # Register tools
    registry = ToolRegistry()
    registry.register(make_search_tool(retriever))
    registry.register(make_summarize_tool())

    return {
        "chunker": chunker,
        "embedder": embedder,
        "store": store,
        "retriever": retriever,
        "evaluator": evaluator,
        "llm": llm,
        "registry": registry,
        "total_chunks": len(all_chunks),
        "doc_count": len(list(doc_dir.glob("*.txt"))),
    }


# Initialize components
engine = init_rag_system()

# -----------------------------------------------------------------------------
# Sidebar: Settings & Metadata Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Agentic RAG")
    st.caption("From Scratch Learning Dashboard")
    st.divider()

    st.subheader("🏷️ Metadata Filter")
    selected_topic = st.selectbox(
        "Topic Pre-Filter",
        options=["All", "basics", "agents", "ml"],
        index=0,
        help="Pre-filter FAISS vector search candidates by document topic tag.",
    )

    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)
    min_score = st.slider("Min Cosine Similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    st.divider()
    st.subheader("🤖 LLM Backend Status")
    if engine["llm"].is_mock:
        st.warning("🟡 Mode: Offline Mock LLM\nSet `OPENAI_API_KEY` in `.env` for real LLM answers.")
    else:
        st.success(f"🟢 Mode: Real OpenAI LLM ({engine['llm'].model})")

    st.caption(f"Indexed Chunks: **{engine['total_chunks']}** across **{engine['doc_count']}** files")


# Build active MetadataFilter from sidebar
filter_spec = None if selected_topic == "All" else MetadataFilter(topic=selected_topic)


# -----------------------------------------------------------------------------
# Main Content Area: Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_inspector, tab_eval, tab_benchmark = st.tabs([
    "💬 Agent Chat",
    "🔍 Chunk Inspector",
    "📊 RAG Evaluator",
    "🏆 Benchmark Suite",
])


# =============================================================================
# TAB 1: Agent Chat with Step-by-Step Thought Inspector
# =============================================================================
with tab_chat:
    st.markdown("### 💬 Ask the Agentic RAG Agent")
    st.caption("The agent will execute a Think ➔ Act ➔ Observe loop using registered tools.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_eval" not in st.session_state:
        st.session_state.last_eval = None
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question (e.g., 'What is Agentic RAG and how does the agent loop work?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agentic query
        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning (Think → Act → Observe)..."):
                # Create isolated agent memory
                agent_memory = Memory(max_history=20)
                agent = AgenticRAG(
                    llm=engine["llm"],
                    tool_registry=engine["registry"],
                    memory=agent_memory,
                    max_iterations=5,
                    verbose=False,
                )

                answer = agent.ask(prompt)
                st.markdown(f"**Final Answer:**\n\n{answer}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Retrieve context & evaluate answer
            retrieved = engine["retriever"].retrieve(
                prompt, top_k=top_k, filter_spec=filter_spec
            )
            st.session_state.last_retrieved = retrieved

            eval_res = engine["evaluator"].evaluate(
                query=prompt, answer=answer, retrieval_results=retrieved
            )
            st.session_state.last_eval = eval_res

        st.rerun()


# =============================================================================
# TAB 2: Chunk Inspector
# =============================================================================
with tab_inspector:
    st.markdown("### 🔍 FAISS Vector Store & Chunk Inspector")
    st.caption("Test semantic search directly and inspect exact retrieved text chunks & metadata.")

    inspect_query = st.text_input(
        "Test Query Search",
        value="What is sliding window chunking?",
        key="inspect_query_input",
    )

    if inspect_query:
        results = engine["retriever"].retrieve(
            inspect_query, top_k=top_k, filter_spec=filter_spec
        )

        st.markdown(f"**Results for:** *'{inspect_query}'* (Filter: `{selected_topic}`)")
        st.divider()

        if not results:
            st.info("No chunks matched the search query and metadata filter.")
        else:
            for r in results:
                topic_name = r.chunk.metadata.get("topic", "N/A")
                source_file = Path(r.chunk.source).name

                st.markdown(f"""
                <div class="chunk-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span><strong>Rank {r.rank}</strong> — <code>{source_file}</code></span>
                        <div>
                            <span class="badge-topic">topic: {topic_name}</span>
                            <span class="badge-score">Similarity: {r.score:.3f}</span>
                        </div>
                    </div>
                    <p style="margin: 0; color: #D1D5DB;">{r.chunk.text}</p>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: RAG Evaluator Scorecard
# =============================================================================
with tab_eval:
    st.markdown("### 📊 RAG Quality Scorecard")
    st.caption("Live evaluation metrics calculated for your last query.")

    eval_data = st.session_state.get("last_eval")

    if eval_data is None:
        st.info("Ask a question in the **💬 Agent Chat** tab to see live RAG quality metrics!")
    else:
        st.markdown(f"**Evaluated Query:** *'{eval_data.query}'*")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Context Relevance", f"{eval_data.context_relevance:.2f}")
        with col2:
            st.metric("Faithfulness", f"{eval_data.faithfulness:.2f}")
        with col3:
            st.metric("Answer Relevance", f"{eval_data.answer_relevance:.2f}")
        with col4:
            st.metric("Context Recall", f"{eval_data.context_recall:.2f}")
        with col5:
            st.metric("Overall Score", f"{eval_data.overall_score:.2f}")

        st.divider()
        st.subheader("Metric Definitions")
        st.markdown("""
        - **Context Relevance**: Measures how relevant the retrieved chunks are to the user query.
        - **Faithfulness**: Measures if the answer facts are strictly grounded in retrieved context.
        - **Answer Relevance**: Measures if the generated answer directly addresses the question.
        - **Context Recall**: Measures coverage of key query terms in retrieved text.
        """)


# =============================================================================
# TAB 4: Benchmark Suite & Architecture
# =============================================================================
with tab_benchmark:
    st.markdown("### 🏆 Automated Evaluation Benchmark")
    st.caption("Run test suite benchmarks to measure overall system quality.")

    if st.button("🚀 Run Full Evaluation Suite (`eval_runner.py`)"):
        with st.spinner("Running evaluation benchmark across test cases..."):
            import subprocess
            cmd = [sys.executable, "-X", "utf8", "eval_runner.py"]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))

            if res.returncode == 0:
                st.success("Benchmark Execution Complete!")
                st.code(res.stdout, language="text")
            else:
                st.error("Benchmark encountered an error:")
                st.code(res.stderr, language="text")

    st.divider()
    st.markdown("### 🗺️ Project Architecture Overview")
    st.code("""
agentic_rag/
├── app.py               ← Streamlit Interactive Web App (This UI)
├── main.py              ← Pipeline Demo & Ingestion
├── eval_runner.py       ← Benchmark Evaluation Suite
├── show_structure.py    ← Terminal Visualizer
│
├── documents/           ← Knowledge Base (.txt files with topic metadata)
│   ├── rag_basics.txt
│   ├── agent_concepts.txt
│   └── ml_concepts.txt
│
├── core/                ← RAG Engine Components
│   ├── chunker.py       ← Sliding window with metadata tag extraction
│   ├── embedder.py      ← MiniLM text embeddings (384-dim)
│   ├── vector_store.py  ← FAISS vector store + metadata pre-filtering
│   ├── retriever.py     ← Cosine similarity top-K search
│   ├── metadata_filter.py ← MetadataFilter rules
│   └── evaluator.py     ← 4-Metric RAG Evaluator
│
├── agent/               ← Agentic Reasoning Brain
│   ├── agent.py         ← Think → Act → Observe → Answer loop
│   ├── tools.py         ← search_knowledge_base + summarize
│   └── memory.py        ← History + scratchpad
│
└── llm/                 ← LLM Provider Interface
    └── llm_client.py    ← OpenAI API client + offline mock
""", language="text")
