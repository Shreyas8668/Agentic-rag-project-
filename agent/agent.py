# =============================================================================
# agent/agent.py
# -----------------------------------------------------------------------------
# CONCEPT: The Agentic Loop
# This is the heart of Agentic RAG. Instead of a single retrieve-then-generate
# step, the agent runs a loop:
#
#   ┌──────────────────────────────────────────────────────┐
#   │                  AGENTIC LOOP                        │
#   │                                                      │
#   │  User Question                                       │
#   │       │                                              │
#   │       ▼                                              │
#   │  ┌─────────┐   THINK: What do I need to know?        │
#   │  │  Plan   │──────────────────────────────────────┐  │
#   │  └─────────┘                                      │  │
#   │       │                                           │  │
#   │       ▼                                           │  │
#   │  ┌─────────┐   ACT: Call a tool (e.g. search)    │  │
#   │  │  Act    │                                      │  │
#   │  └─────────┘                                      │  │
#   │       │                                           │  │
#   │       ▼                                           │  │
#   │  ┌──────────┐  OBSERVE: Read tool result          │  │
#   │  │ Observe  │──► enough info? No ──────────────────┘  │
#   │  └──────────┘          Yes                           │
#   │       │                                              │
#   │       ▼                                              │
#   │  ┌──────────┐  ANSWER: Generate final response       │
#   │  │  Answer  │                                        │
#   │  └──────────┘                                        │
#   └──────────────────────────────────────────────────────┘
#
# The loop has a maximum iteration limit to prevent infinite loops.
# =============================================================================

from __future__ import annotations
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from .memory import Memory
from .tools import ToolRegistry
from llm.llm_client import LLMClient

console = Console()

# ANSI-style colors mapped to rich styles
STEP_STYLES = {
    "think":   "bold cyan",
    "act":     "bold yellow",
    "observe": "bold magenta",
    "answer":  "bold green",
    "error":   "bold red",
}


class AgenticRAG:
    """
    The Agentic RAG agent.

    Runs a Think → Act → Observe loop, using tools to gather information
    before producing a final grounded answer.

    Parameters
    ----------
    llm           : LLMClient     – the language model backend
    tool_registry : ToolRegistry  – all available tools
    memory        : Memory        – conversation + scratchpad memory
    max_iterations: int           – safety limit on loop iterations
    verbose       : bool          – print each step to the terminal
    """

    SYSTEM_PROMPT_TEMPLATE = """\
You are an intelligent Agentic RAG assistant. Your goal is to answer the user's
question accurately by searching the knowledge base when needed.

{tool_descriptions}

INSTRUCTIONS:
- First, THINK about what information you need.
- If you need to look something up, call a tool using EXACTLY this format on its own line:
    TOOL: <tool_name> | QUERY: <your search query>
- After receiving tool results, THINK again: do you have enough information?
- If yes, write your final answer prefixed with: FINAL ANSWER:
- If no, call another tool (up to {max_iter} total tool calls).
- Always base your final answer on the retrieved evidence, not on prior knowledge.
- Be concise and educational — this is a learning system.
"""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        memory: Optional[Memory] = None,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.memory = memory or Memory()
        self.max_iterations = max_iterations
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """
        Ask the agent a question and get a grounded answer.

        This method drives the full Think → Act → Observe loop.
        """
        self._print_header(question)
        self.memory.add_user_message(question)
        self.memory.clear_scratchpad()

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self.tools.get_tool_descriptions(),
            max_iter=self.max_iterations,
        )

        answer = "(No answer generated)"

        for iteration in range(1, self.max_iterations + 1):
            self._log("think", f"Iteration {iteration}/{self.max_iterations} — asking the LLM...")

            # ── THINK: ask the LLM what to do next ────────────────────
            llm_response = self.llm.chat(
                messages=self.memory.get_history(),
                system=system_prompt,
            )
            self._log("think", f"LLM says:\n{llm_response}")
            self.memory.add_assistant_message(llm_response)

            # ── Check for FINAL ANSWER ─────────────────────────────────
            if "FINAL ANSWER:" in llm_response:
                answer = llm_response.split("FINAL ANSWER:", 1)[1].strip()
                self._log("answer", f"\n{answer}")
                break

            # ── ACT: parse and execute tool call ──────────────────────
            tool_result = self.tools.parse_and_execute(llm_response)

            if tool_result is None:
                # No tool call found — treat the whole response as the answer
                answer = llm_response.strip()
                self._log("answer", f"(No tool called — treating as final)\n{answer}")
                break

            self._log("act", f"Tool called. Result:\n{tool_result[:600]}")

            # ── OBSERVE: feed result back into memory ─────────────────
            observation = f"OBSERVATION from tool:\n{tool_result}"
            self.memory.add_tool_result(observation)
            self.memory.note(f"Iter {iteration}: retrieved information")
            self._log("observe", "Stored tool result in memory. Continuing loop...")

        else:
            # Loop exhausted without a final answer
            answer = (
                "I searched the knowledge base but could not find a definitive answer. "
                "Please try rephrasing your question."
            )
            self._log("error", "Max iterations reached without FINAL ANSWER.")

        self.memory.add_assistant_message(f"FINAL ANSWER: {answer}")
        self.memory.clear_scratchpad()
        return answer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, step: str, message: str) -> None:
        """Print a colored step label + message to the terminal."""
        if not self.verbose:
            return
        style = STEP_STYLES.get(step, "white")
        label = f"[{step.upper()}]"
        console.print(f"\n[{style}]{label}[/{style}] {message}")

    def _print_header(self, question: str) -> None:
        """Print a decorative header for each new question."""
        if not self.verbose:
            return
        console.print(
            Panel(
                Text(f"❓  {question}", style="bold white"),
                title="[bold blue]Agentic RAG — New Query[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 4),
            )
        )
