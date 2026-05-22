"""Application configuration loaded from environment variables.

Uses pydantic-settings for type-safe configuration with validation.
All secrets and connection details come from .env file (or environment).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupabaseSettings(BaseSettings):
    """Supabase connection configuration."""

    url: str = Field(..., description="Supabase project URL")
    anon_key: SecretStr = Field(..., description="Anonymous/publishable key")
    service_role_key: SecretStr = Field(..., description="Service role key (backend only)")

    db_host: str = Field(..., description="Postgres host")
    db_port: int = Field(default=6543, description="Postgres port (pooler)")
    db_user: str = Field(..., description="Postgres user")
    db_password: SecretStr = Field(..., description="Postgres password")
    db_name: str = Field(default="postgres", description="Database name")

    model_config = SettingsConfigDict(env_prefix="SUPABASE_", extra="ignore")

    @computed_field  # type: ignore[misc]
    @property
    def connection_string(self) -> str:
        """Build PostgreSQL connection string for direct DB access."""
        password = self.db_password.get_secret_value()
        return (
            f"postgresql://{self.db_user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy-compatible URL using psycopg3."""
        password = self.db_password.get_secret_value()
        return (
            f"postgresql+psycopg://{self.db_user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


class AnthropicSettings(BaseSettings):
    """Anthropic API configuration for news intelligence layer."""

    api_key: SecretStr | None = Field(default=None, description="Anthropic API key")
    model_haiku: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Haiku model for first-pass extraction",
    )
    model_sonnet: str = Field(
        default="claude-sonnet-4-6",
        description="Sonnet model for complex analysis",
    )

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", extra="ignore")


class DataLoadSettings(BaseSettings):
    """Data ingestion configuration."""

    sackmann_atp_repo: str = Field(
        default="https://github.com/JeffSackmann/tennis_atp.git",
        description="Sackmann ATP repository URL",
    )
    sackmann_wta_repo: str = Field(
        default="https://github.com/JeffSackmann/tennis_wta.git",
        description="Sackmann WTA repository URL",
    )
    load_start_year: int = Field(default=2000, description="Start year for data load")
    load_end_year: int = Field(default=2024, description="End year for data load")

    model_config = SettingsConfigDict(env_prefix="DATA_LOAD_", extra="ignore")


class AppSettings(BaseSettings):
    """Application-level settings."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    environment: Literal["development", "staging", "production"] = Field(default="development")

    # Project paths
    project_root: Path = Field(default=Path(__file__).parent.parent.parent.resolve())
    data_dir: Path = Field(default=Path("data"))
    logs_dir: Path = Field(default=Path("logs"))

    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    """Root settings container."""

    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)  # type: ignore[arg-type]
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    data_load: DataLoadSettings = Field(default_factory=DataLoadSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached singleton settings instance.

    Cached so we don't re-read .env on every call.
    Use this in application code; tests can override via dependency injection.
    """
    return Settings()  # type: ignore[call-arg]
