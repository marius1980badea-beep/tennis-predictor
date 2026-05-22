"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupabaseSettings(BaseSettings):
    url: str = Field(...)
    anon_key: SecretStr = Field(...)
    service_role_key: SecretStr = Field(...)
    db_host: str = Field(...)
    db_port: int = Field(default=6543)
    db_user: str = Field(...)
    db_password: SecretStr = Field(...)
    db_name: str = Field(default="postgres")

    model_config = SettingsConfigDict(
        env_prefix="SUPABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def connection_string(self) -> str:
        password = self.db_password.get_secret_value()
        return f"postgresql://{self.db_user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @computed_field
    @property
    def sqlalchemy_url(self) -> str:
        password = self.db_password.get_secret_value()
        return f"postgresql+psycopg://{self.db_user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"


class AnthropicSettings(BaseSettings):
    api_key: SecretStr | None = Field(default=None)
    model_haiku: str = Field(default="claude-haiku-4-5-20251001")
    model_sonnet: str = Field(default="claude-sonnet-4-6")

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class DataLoadSettings(BaseSettings):
    sackmann_atp_repo: str = Field(default="https://github.com/JeffSackmann/tennis_atp.git")
    sackmann_wta_repo: str = Field(default="https://github.com/JeffSackmann/tennis_wta.git")
    load_start_year: int = Field(default=2000)
    load_end_year: int = Field(default=2024)

    model_config = SettingsConfigDict(
        env_prefix="DATA_LOAD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppSettings(BaseSettings):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    project_root: Path = Field(default=Path(__file__).parent.parent.parent.resolve())
    data_dir: Path = Field(default=Path("data"))
    logs_dir: Path = Field(default=Path("logs"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(BaseSettings):
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    data_load: DataLoadSettings = Field(default_factory=DataLoadSettings)
    app: AppSettings = Field(default_factory=AppSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
