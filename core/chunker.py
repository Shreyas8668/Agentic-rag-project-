# =============================================================================
# core/chunker.py
# -----------------------------------------------------------------------------
# CONCEPT: Chunking
# Large documents cannot be embedded as a whole — the embedding model has a
# token limit, and huge chunks produce blurry, averaged-out vectors that match
# poorly during retrieval.
#
# Strategy used here: SLIDING WINDOW with OVERLAP
#
#   ┌──────────────────── full document ────────────────────────┐
#   │ chunk_0 │                                                  │
#   │      chunk_1 │                                            │
#   │           chunk_2 │                                       │
#   └───────────────────────────────────────────────────────────┘
#        ←chunk_size→
#             ←overlap→
#
# The overlap makes sure that a sentence at the end of one chunk also appears
# at the start of the next — so we never lose context at boundaries.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """A single piece of text cut from a document."""
    text: str                        # The actual text content
    source: str                      # Which file it came from
    chunk_id: int                    # Index within the source file
    start_char: int                  # Character offset in the original document
    metadata: dict = field(default_factory=dict)  # Any extra info

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source='{self.source}', text='{preview}...')"


class Chunker:
    """
    Splits raw text into overlapping chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk.
    overlap : int
        Number of characters to repeat between consecutive chunks.
        Must be less than chunk_size.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self, text: str, source: str = "unknown", extra_metadata: dict | None = None
    ) -> List[Chunk]:
        """
        Split `text` into overlapping Chunk objects with extracted metadata.
        """
        # Parse document-level header tags (e.g. # TOPIC: agents, # DATE: 2024-01-01)
        doc_metadata = self._extract_header_metadata(text)
        if extra_metadata:
            doc_metadata.update(extra_metadata)
        doc_metadata.setdefault("source", source)

        cleaned_text = self._clean(text)
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        chunk_id = 0

        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))
            chunk_text = cleaned_text[start:end].strip()

            if chunk_text:
                chunk_meta = doc_metadata.copy()
                chunk_meta["chunk_id"] = chunk_id

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        chunk_id=chunk_id,
                        start_char=start,
                        metadata=chunk_meta,
                    )
                )
                chunk_id += 1

            if end == len(cleaned_text):
                break
            start += step

        return chunks

    def chunk_file(self, filepath: str, extra_metadata: dict | None = None) -> List[Chunk]:
        """Read a file from disk and chunk its contents with metadata."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        file_meta = {"source": filepath}
        if extra_metadata:
            file_meta.update(extra_metadata)

        return self.chunk_text(text, source=filepath, extra_metadata=file_meta)

    @staticmethod
    def _extract_header_metadata(text: str) -> dict:
        """Extract key-value tags defined in leading lines like `# KEY: value`."""
        import re
        metadata = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                match = re.match(r"^#\s*([A-Z_]+):\s*(.+)$", line, re.IGNORECASE)
                if match:
                    key = match.group(1).lower()
                    val = match.group(2).strip()
                    metadata[key] = val
            elif line:
                # Stop parsing at first non-header, non-empty line
                break
        return metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace while preserving paragraph structure."""
        import re
        # Collapse 3+ newlines into 2 (keep paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces/tabs into one
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
