"""Data loaders for various sources.

Note: Sub-modules are imported lazily so that unit tests can test
pure helper functions without needing all heavy dependencies installed.
"""

from __future__ import annotations

__all__ = ["SackmannLoader"]


def __getattr__(name: str):
    """Lazy import on attribute access. Triggered only when needed."""
    if name == "SackmannLoader":
        from tennis_predictor.data.loaders.sackmann import SackmannLoader

        return SackmannLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
