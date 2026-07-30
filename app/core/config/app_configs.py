"""
Application identity and FastAPI server settings.

Defaults live in yaml/app.yaml and are directly editable there. Values set in
.env (or the process environment) override the YAML defaults — reserve .env
for secrets and per-environment overrides, not baseline configuration.
"""

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class AppSettings(BaseSettings):
    """Application identity and FastAPI server configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(..., description="Application name")
    app_version: str = Field(..., description="Application version")
    app_environment: str = Field(
        ..., description="Application environment (development, staging, production)"
    )
    debug: bool = Field(..., description="Debug mode")
    log_level: str = Field(..., description="Logging level")
    fastapi_host: str = Field(..., description="FastAPI host")
    fastapi_port: int = Field(..., description="FastAPI port")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env vars and .env override YAML defaults; YAML is the baseline source."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls, yaml_file=CONFIG_YAML_DIR / "app.yaml"
            ),
            file_secret_settings,
        )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is acceptable."""
        valid_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("app_environment")
    @classmethod
    def validate_app_environment(cls, v: str) -> str:
        """Validate application environment value."""
        normalized = v.lower().strip()
        valid_envs = {"development", "staging", "production"}
        if normalized not in valid_envs:
            raise ValueError(
                f"app_environment must be one of {sorted(valid_envs)}, got '{v}'"
            )
        return normalized
