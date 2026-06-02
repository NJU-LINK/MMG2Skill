"""Shared text helpers for the reviser module."""

from __future__ import annotations


def truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* chars, appending a marker if cut.

    ``limit <= 0`` disables truncation (returns *text* unchanged).
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...[truncated]"
