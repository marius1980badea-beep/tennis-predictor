"""Direct PostgreSQL connection via SQLAlchemy.

Used for bulk operations (COPY, large INSERTs) where Supabase REST API
would be too slow. Uses psycopg3 driver via SQLAlchemy 2.x.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from tennis_predictor.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Get singleton SQLAlchemy engine.

    Uses Supabase connection pooler (port 6543) for efficient connection reuse.
    Disables prepared statements (incompatible with pgBouncer transaction mode).
    """
    settings = get_settings()
    return create_engine(
        settings.supabase.sqlalchemy_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        # CRITICAL: disable prepared statements for Supabase pooler compatibility
        # See: https://www.psycopg.org/psycopg3/docs/advanced/prepare.html
        connect_args={"prepare_threshold": None},
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Get singleton session factory."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager for database sessions.

    Automatically commits on success, rolls back on exception.

    Usage:
        with get_session() as session:
            session.execute(...)
            # Auto-commits when block exits successfully
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
