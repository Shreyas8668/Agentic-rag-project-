# =============================================================================
# agent/memory.py
# -----------------------------------------------------------------------------
# CONCEPT: Agent Memory
# A stateless agent can only answer based on the current message — it forgets
# everything from previous turns. Memory gives the agent continuity.
#
# Types of memory in Agentic systems:
# ┌─────────────────────────────────────────────────────────────┐
# │ SHORT-TERM  (this file)                                     │
# │  • Conversation history  – the Q&A so far                  │
# │  • Scratchpad            – working notes during agent loop  │
# ├─────────────────────────────────────────────────────────────┤
# │ LONG-TERM   (future extension)                              │
# │  • Persistent vector store of past conversations            │
# │  • User preferences, learned facts                          │
# └─────────────────────────────────────────────────────────────┘
#
# The scratchpad is a temporary notepad the agent writes to during one
# reasoning loop (Think → Act → Observe). It is cleared after each final answer.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal


# Type alias for message roles (matches OpenAI's convention)
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single message in the conversation."""
    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert to the dict format expected by LLM APIs."""
        return {"role": self.role, "content": self.content}


class Memory:
    """
    Manages short-term memory for one agent session.

    Attributes
    ----------
    max_history : int
        Keep only the last N messages (older ones are dropped to save tokens).
    scratchpad  : list[str]
        Temporary observations written during the current agent loop.
        Cleared after each final answer.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._history: List[Message] = []
        self.scratchpad: List[str] = []  # agent's working notes

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        """Record a message from the human user."""
        self._history.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Record a response from the assistant / agent."""
        self._history.append(Message(role="assistant", content=content))
        self._trim()

    def add_tool_result(self, content: str) -> None:
        """Record the output of a tool call (shown to the agent as observation)."""
        self._history.append(Message(role="tool", content=content))
        self._trim()

    def get_history(self) -> List[dict]:
        """Return the history as a list of dicts for the LLM API."""
        return [m.to_dict() for m in self._history]

    # ------------------------------------------------------------------
    # Scratchpad (per-loop working notes)
    # ------------------------------------------------------------------

    def note(self, observation: str) -> None:
        """Write an observation to the scratchpad."""
        self.scratchpad.append(observation)

    def get_scratchpad(self) -> str:
        """Return all scratchpad notes as a single formatted string."""
        if not self.scratchpad:
            return "(empty)"
        return "\n".join(f"• {note}" for note in self.scratchpad)

    def clear_scratchpad(self) -> None:
        """Clear the scratchpad after a final answer is given."""
        self.scratchpad.clear()

    def clear_all(self) -> None:
        """Reset everything — start a fresh session."""
        self._history.clear()
        self.scratchpad.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Drop oldest messages when history exceeds max_history."""
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def __repr__(self) -> str:
        return (
            f"Memory(messages={len(self._history)}, "
            f"scratchpad_notes={len(self.scratchpad)})"
        )
