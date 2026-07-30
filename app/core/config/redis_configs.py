"""
Redis connection, token-counter fast path, and cache settings.

Defaults live in yaml/redis.yaml and are directly editable there.
redis_password is a secret and is intentionally absent from the YAML defaults —
it is sourced from .env only. Values set in .env (or the process environment)
override the YAML defaults for every other field.
"""

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class RedisSettings(BaseSettings):
    """Redis connection, token-counter fast path, and cache configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    redis_host: str = Field(..., description="Redis host")
    redis_port: int = Field(..., description="Redis port")
    redis_db: int = Field(..., description="Redis database number")
    redis_password: str | None = Field(
        default=None, description="Redis password (secret — set via .env only)"
    )
    redis_max_connections: int = Field(..., description="Redis max connections")

    redis_token_counter_ttl_secs: int = Field(
        ...,
        description=(
            "TTL for Redis token counter keys (seconds). Reconciler re-seeds on expiry."
        ),
    )
    redis_token_counter_db: int = Field(
        ...,
        description=(
            "Redis logical DB index for token counters "
            "(isolates from rate limiter DB 0)"
        ),
    )
    redis_token_counter_max_connections: int = Field(
        ...,
        description="Dedicated Redis connection limit for token counter operations",
    )

    cache_enabled: bool = Field(..., description="Enable response caching")
    cache_ttl_seconds: int = Field(..., description="Cache TTL in seconds")

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
                settings_cls, yaml_file=CONFIG_YAML_DIR / "redis.yaml"
            ),
            file_secret_settings,
        )

    @field_validator(
        "redis_token_counter_ttl_secs",
        "redis_token_counter_max_connections",
    )
    @classmethod
    def validate_positive_redis_integers(cls, v: int) -> int:
        """Validate that these Redis settings are positive."""
        if v <= 0:
            raise ValueError("Setting must be greater than 0")
        return v

    @property
    def redis_url(self) -> str:
        """Construct Redis URL (DB 0 — rate limiting / general cache)."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@"
                f"{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def redis_token_counter_url(self) -> str:
        """Construct Redis URL for the token counter fast path on isolated DB 1."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@"
                f"{self.redis_host}:{self.redis_port}/{self.redis_token_counter_db}"
            )
        return (
            f"redis://{self.redis_host}:{self.redis_port}/{self.redis_token_counter_db}"
        )
