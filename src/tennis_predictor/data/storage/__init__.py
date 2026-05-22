"""Database storage and Supabase client modules.

Note: Imports inside functions are intentional - keeps unit tests
of pure helpers (sackmann.py transformations) able to import the
loader module without requiring sqlalchemy/supabase to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

    from sqlalchemy import Engine
    from sqlalchemy.orm import Session
    from supabase import Client


def get_engine() -> "Engine":
    """Get singleton SQLAlchemy engine (lazy import)."""
    from tennis_predictor.data.storage.db import get_engine as _get_engine

    return _get_engine()


def get_session() -> "AbstractContextManager[Session]":
    """Get a database session context manager (lazy import)."""
    from tennis_predictor.data.storage.db import get_session as _get_session

    return _get_session()


def get_supabase_client() -> "Client":
    """Get the Supabase client (lazy import)."""
    from tennis_predictor.data.storage.supabase_client import (
        get_supabase_client as _get_supabase_client,
    )

    return _get_supabase_client()


__all__ = ["get_engine", "get_session", "get_supabase_client"]
