"""
Rate limiting traffic-policy settings.

Defaults live in yaml/rate_limit.yaml and are directly editable there. Values
set in .env (or the process environment) override the YAML defaults for
per-environment tuning.

Scope note: these are traffic-policy thresholds (requests/minute per
endpoint), not Redis connection config. Redis is only the storage backend the
`limits` library happens to use — see redis_configs.py for connection
settings. Same reasoning that already kept cb_redis_failure_threshold /
cb_redis_recovery_timeout out of redis_configs.py and into
resiliency_configs.py instead.
"""

from ipaddress import ip_network

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class RateLimitSettings(BaseSettings):
    """Rate limiting traffic-policy configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    rate_limit_requests_per_minute: int = Field(
        ..., description="Max requests per minute per user"
    )
    rate_limit_window_seconds: int = Field(
        ..., description="Rate limit window in seconds"
    )

    rate_limit_token_generate_per_minute: int = Field(
        ..., description="Max token generate requests per minute per IP"
    )
    rate_limit_token_refresh_per_minute: int = Field(
        ..., description="Max token refresh requests per minute per IP"
    )

    rate_limit_token_acquire_per_minute: int = Field(
        ..., description="Max /acquire requests per minute per verified client IP"
    )

    # Each proxy appends the IP it received from to the RIGHT of
    # X-Forwarded-For, so the real client IP is read this many positions from
    # the right; entries to the left are client-supplied and untrusted.
    # Default 0 trusts only the TCP peer. Nonzero requires an explicit list
    # of proxy networks below; a hop count alone never grants header trust.
    rate_limit_trusted_proxy_hops: int = Field(
        ...,
        ge=0,
        description=(
            "Trusted reverse-proxy hops; client IP is read this many "
            "positions from the right of X-Forwarded-For"
        ),
    )
    rate_limit_trusted_proxy_networks: list[str] = Field(
        default_factory=list,
        description="TCP peer networks authorized to supply X-Forwarded-For",
    )

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
                settings_cls, yaml_file=CONFIG_YAML_DIR / "rate_limit.yaml"
            ),
            file_secret_settings,
        )

    @field_validator(
        "rate_limit_requests_per_minute",
        "rate_limit_window_seconds",
        "rate_limit_token_generate_per_minute",
        "rate_limit_token_refresh_per_minute",
        "rate_limit_token_acquire_per_minute",
    )
    @classmethod
    def validate_positive_rate_limit_integers(cls, v: int) -> int:
        """
        Validate that these rate-limit thresholds are positive.

        A zero or negative limit is not "unlimited" to the `limits` library —
        it either blocks every request or raises at request time deep inside a
        third-party rate limiter. Reject it at startup instead.
        """
        if v <= 0:
            raise ValueError("Setting must be greater than 0")
        return v

    @field_validator("rate_limit_trusted_proxy_networks")
    @classmethod
    def validate_trusted_proxy_networks(cls, networks: list[str]) -> list[str]:
        """Reject invalid proxy network entries at configuration load time."""
        for network in networks:
            ip_network(network, strict=False)
        return networks
