# =============================================================================
# agent/tools.py
# -----------------------------------------------------------------------------
# CONCEPT: Tools
# Tools are the hands of the agent — they let it interact with the world beyond
# just generating text.  Each tool is:
#   • A Python function that does real work
#   • A schema (name + description) that the agent reads to know WHAT to call
#
# Our agent uses a simple text-based tool calling protocol:
#   Agent outputs:  TOOL: search_knowledge_base | QUERY: What is RAG?
#   We parse this → call the function → return the result as an observation.
#
# Real-world tools include: web search, code execution, database queries,
# email/calendar APIs, etc.
# =============================================================================

from __future__ import annotations
import re
import textwrap
from typing import Callable, Dict, List, Optional


class Tool:
    """Wraps a Python function as an agent-callable tool."""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self._func = func

    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        result = self._func(**kwargs)
        return str(result)

    def schema(self) -> str:
        """Human-readable description for the agent's system prompt."""
        return f"  • {self.name}: {self.description}"


class ToolRegistry:
    """
    Holds all available tools and parses tool-call strings from the agent.

    The agent signals a tool call by outputting a line like:
        TOOL: <tool_name> | QUERY: <argument>

    The registry:
      1. Detects this pattern
      2. Looks up the right tool
      3. Executes it
      4. Returns the observation string
    """

    # Regex to detect a tool call in the agent's output
    TOOL_CALL_PATTERN = re.compile(
        r"TOOL:\s*(?P<tool>\w+)\s*\|\s*QUERY:\s*(?P<query>.+)", re.IGNORECASE
    )

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name.lower()] = tool

    def get_tool_descriptions(self) -> str:
        """Return a formatted list of all tools for injection into the system prompt."""
        lines = ["Available tools (use exactly this format to call them):"]
        lines.append('  TOOL: <tool_name> | QUERY: <your question or input>')
        lines.append("")
        for tool in self._tools.values():
            lines.append(tool.schema())
        return "\n".join(lines)

    def parse_and_execute(self, agent_output: str) -> Optional[str]:
        """
        Check if `agent_output` contains a tool call.
        If yes, run the tool and return the result.
        If no tool call found, return None.
        """
        match = self.TOOL_CALL_PATTERN.search(agent_output)
        if not match:
            return None

        tool_name = match.group("tool").lower()
        query = match.group("query").strip()

        tool = self._tools.get(tool_name)
        if tool is None:
            return f"[Error] Unknown tool '{tool_name}'. Available: {list(self._tools.keys())}"

        try:
            return tool.run(query=query)
        except Exception as e:
            return f"[Error] Tool '{tool_name}' raised an exception: {e}"

    def list_tools(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())


# =============================================================================
# Built-in tool factory functions
# These are not Tool instances themselves — they return Tool instances so we
# can inject dependencies (like the retriever) at construction time.
# =============================================================================

def make_search_tool(retriever) -> Tool:
    """
    Create the 'search_knowledge_base' tool backed by our retriever.
    Supports optional topic metadata filtering: QUERY: <text> | TOPIC: <topic>
    """
    def _search(query: str) -> str:
        from core.metadata_filter import MetadataFilter

        topic_filter = None
        clean_query = query

        # Parse optional | TOPIC: <topic_name>
        if "| TOPIC:" in query.upper():
            parts = re.split(r"\|\s*TOPIC:\s*", query, flags=re.IGNORECASE)
            clean_query = parts[0].strip()
            topic_name = parts[1].strip()
            topic_filter = MetadataFilter(topic=topic_name)

        results = retriever.retrieve(clean_query, top_k=3, filter_spec=topic_filter)
        if not results:
            filter_note = f" (filtered by topic='{topic_filter.topic}')" if topic_filter else ""
            return f"No relevant information found{filter_note}."

        formatted = []
        for r in results:
            topic_tag = f" [Topic: {r.chunk.metadata.get('topic')}]" if r.chunk.metadata.get('topic') else ""
            formatted.append(
                f"[Score: {r.score:.3f}{topic_tag}] {r.chunk.text[:400]}"
            )
        return "\n\n".join(formatted)

    return Tool(
        name="search_knowledge_base",
        description=(
            "Search the knowledge base for facts. "
            "Optional metadata filtering syntax: TOOL: search_knowledge_base | QUERY: <question> | TOPIC: <basics|agents|ml>"
        ),
        func=_search,
    )


def make_summarize_tool() -> Tool:
    """
    Create a 'summarize' tool that condenses long text.
    In a real system this would call an LLM with a summarisation prompt.
    Here we do a simple extractive summarisation for offline use.
    """
    def _summarize(query: str) -> str:
        # Simple extractive: return first 3 sentences
        sentences = re.split(r"(?<=[.!?])\s+", query.strip())
        summary = " ".join(sentences[:3])
        return f"[Summary] {summary}"

    return Tool(
        name="summarize",
        description=(
            "Summarize a long piece of text into a short paragraph. "
            "Pass the full text as the QUERY."
        ),
        func=_summarize,
    )
