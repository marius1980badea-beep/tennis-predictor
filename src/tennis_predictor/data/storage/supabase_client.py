"""Supabase client factory.

Provides a singleton Supabase client using the service role key.
Service role bypasses RLS - intended for backend operations only.
NEVER expose this client or key to frontend code.
"""

from functools import lru_cache

from supabase import Client, create_client

from tennis_predictor.config import get_settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Get singleton Supabase client with service role privileges.

    Uses service_role_key which bypasses Row Level Security.
    This is intended for backend operations - never use in user-facing code.

    Returns:
        Configured Supabase client.
    """
    settings = get_settings()
    return create_client(
        supabase_url=settings.supabase.url,
        supabase_key=settings.supabase.service_role_key.get_secret_value(),
    )
