# =============================================================================
# core/metadata_filter.py
# -----------------------------------------------------------------------------
# CONCEPT: Metadata Filtering
# Metadata filtering restricts vector search candidate space using structured tags
# attached to chunks (such as topic, source file, section header, or date).
#
# Filtering happens BEFORE vector ranking (pre-filtering):
#   1. Filter all candidate chunks by metadata rules → valid subset
#   2. Perform FAISS vector search ONLY over the valid subset
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class MetadataFilter:
    """
    Specifies filtering rules for vector retrieval.

    Attributes
    ----------
    topic : Optional[str]
        Exact match on topic tag (e.g. 'basics', 'agents', 'ml')
    source : Optional[str]
        Exact match or substring match on source filename
    date_from : Optional[str]
        ISO date string (YYYY-MM-DD), inclusive lower bound
    date_to : Optional[str]
        ISO date string (YYYY-MM-DD), inclusive upper bound
    custom : Dict[str, Any]
        Arbitrary additional exact match key-value pairs
    """
    topic: Optional[str] = None
    source: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def matches(self, metadata: dict) -> bool:
        """
        Check if a chunk's metadata dictionary satisfies this filter.

        Returns True if ALL specified conditions pass (AND logic).
        """
        if not metadata:
            # If a filter is specified but chunk has no metadata, it fails
            return not self.has_rules()

        # 1. Topic filter (exact match, case-insensitive)
        if self.topic is not None:
            chunk_topic = str(metadata.get("topic", "")).lower()
            if chunk_topic != self.topic.lower():
                return False

        # 2. Source filter (substring or exact)
        if self.source is not None:
            chunk_source = str(metadata.get("source", "")).lower()
            if self.source.lower() not in chunk_source:
                return False

        # 3. Date range filter
        chunk_date = metadata.get("date", "")
        if self.date_from is not None and chunk_date:
            if chunk_date < self.date_from:
                return False
        if self.date_to is not None and chunk_date:
            if chunk_date > self.date_to:
                return False

        # 4. Custom key-value matches
        for key, val in self.custom.items():
            if metadata.get(key) != val:
                return False

        return True

    def has_rules(self) -> bool:
        """Returns True if any filtering criteria are active."""
        return (
            self.topic is not None
            or self.source is not None
            or self.date_from is not None
            or self.date_to is not None
            or bool(self.custom)
        )

    def __repr__(self) -> str:
        rules = []
        if self.topic: rules.append(f"topic='{self.topic}'")
        if self.source: rules.append(f"source='{self.source}'")
        if self.date_from: rules.append(f"date>='{self.date_from}'")
        if self.date_to: rules.append(f"date<='{self.date_to}'")
        for k, v in self.custom.items(): rules.append(f"{k}='{v}'")
        
        rules_str = ", ".join(rules) if rules else "none"
        return f"MetadataFilter({rules_str})"
