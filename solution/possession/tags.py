"""Helpers for Smart Tagging events."""


def norm(text) -> str:
    """Normalise a tag string for case-insensitive matching."""
    return (text or "").strip().lower()
