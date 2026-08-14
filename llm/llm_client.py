# =============================================================================
# llm/llm_client.py
# -----------------------------------------------------------------------------
# CONCEPT: LLM Abstraction
# We wrap the actual LLM call behind a thin interface so the rest of the code
# never imports 'openai' directly. This lets you swap providers easily:
#
#   OpenAI GPT-4o-mini  ←── LLMClient ──→  Ollama (local)
#                                      ──→  Anthropic Claude
#                                      ──→  MockLLM (offline, for testing)
#
# The client always receives:
#   • messages: List[dict]  – the conversation so far
#   • system  : str         – the system prompt (agent instructions)
# And always returns:
#   • str – the assistant's response text
# =============================================================================

from __future__ import annotations
import os
from typing import List


class LLMClient:
    """
    Thin wrapper around an LLM provider.

    Priority order:
      1. If OPENAI_API_KEY is set in the environment → use OpenAI
      2. Otherwise → use MockLLM (offline, returns template answers)

    Parameters
    ----------
    model : str   – OpenAI model name (ignored in mock mode)
    """

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY", "")
        self._use_mock = not api_key or api_key.startswith("sk-...") or api_key == ""

        if self._use_mock:
            self._backend = _MockLLM()
        else:
            self._backend = _OpenAIBackend(model=self.model, api_key=api_key)

    def chat(self, messages: List[dict], system: str = "") -> str:
        """
        Send a list of messages to the LLM and return its response.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...} dicts
        system   : system prompt string (prepended as system message)

        Returns
        -------
        str – the LLM's response text
        """
        return self._backend.chat(messages=messages, system=system)

    @property
    def is_mock(self) -> bool:
        """True if running in offline mock mode."""
        return self._use_mock

    def __repr__(self) -> str:
        mode = "mock (offline)" if self._use_mock else f"OpenAI ({self.model})"
        return f"LLMClient(backend={mode})"


# =============================================================================
# Backend implementations (private — not exported)
# =============================================================================

class _OpenAIBackend:
    """Calls the real OpenAI Chat Completions API (compatible with openai>=2.x)."""

    def __init__(self, model: str, api_key: str):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except Exception as e:
            raise ImportError(f"Could not initialise OpenAI client: {e}")
        self.model = model

    def chat(self, messages: List[dict], system: str = "") -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=0.2,    # low temp → more factual, less creative
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


class _MockLLM:
    """
    Offline mock LLM for learning / testing without an API key.
    It looks at the last user message and tries to give a sensible canned reply.
    It also correctly simulates the agent's tool-calling format.
    """

    # Keyword → canned tool call that the agent "decides" to make
    _TRIGGERS = {
        "what is": "search_knowledge_base",
        "explain":  "search_knowledge_base",
        "how does": "search_knowledge_base",
        "define":   "search_knowledge_base",
        "tell me":  "search_knowledge_base",
        "summarize": "summarize",
    }

    def chat(self, messages: List[dict], system: str = "") -> str:
        # ── Scan the full message history to understand context ────────
        last_user_question = ""    # the original human question
        last_observation   = ""    # the most recent tool observation
        tool_was_called    = False # has any tool been called yet?

        for m in messages:
            if m["role"] == "user":
                last_user_question = m["content"]
            elif m["role"] == "assistant" and "TOOL:" in m["content"]:
                tool_was_called = True
            elif m["role"] == "tool":
                last_observation = m["content"]  # keep latest observation

        # ── If we have a fresh tool observation → generate FINAL ANSWER ─
        if last_observation:
            return self._final_answer(last_observation, last_user_question)

        # ── No observation yet → decide to call a tool ─────────────────
        lower = last_user_question.lower()
        for trigger, tool_name in self._TRIGGERS.items():
            if trigger in lower:
                # Use the original clean question as the search query
                query = last_user_question.strip().rstrip("?").strip()
                return (
                    f"I need to look this up in the knowledge base.\n"
                    f"TOOL: {tool_name} | QUERY: {query}"
                )

        # ── Fallback: answer directly without a tool ───────────────────
        return (
            "FINAL ANSWER: Agentic RAG combines retrieval systems with "
            "autonomous agent loops so the model can decide when and how to search "
            "for information before generating an answer. Each iteration follows "
            "a Think → Act → Observe cycle until enough evidence is gathered."
        )

    @staticmethod
    def _final_answer(context: str, question: str = "") -> str:
        """Generate a template final answer from retrieved context."""
        # Extract the most informative part of the observation
        # (strip the "OBSERVATION from tool:" header if present)
        body = context
        if "OBSERVATION from tool:" in context:
            body = context.split("OBSERVATION from tool:", 1)[1].strip()

        snippet = body[:300].replace("\n", " ").strip()
        q_note = f' to "{question.strip()}"' if question else ""

        return (
            f"FINAL ANSWER: Based on the retrieved knowledge base context{q_note}:\n\n"
            f"{snippet}...\n\n"
            f"This information was retrieved via semantic search. "
            f"Set OPENAI_API_KEY in your .env file to get a full natural-language answer from a real LLM."
        )
