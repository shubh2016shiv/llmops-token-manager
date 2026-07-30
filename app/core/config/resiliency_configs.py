"""
Circuit breaker and backpressure settings.

Defaults live in yaml/resiliency.yaml and are directly editable there. Values
set in .env (or the process environment) override the YAML defaults for
per-environment tuning.

Scope note: "Gateway task settings" (gateway_result_ttl_seconds,
gateway_worker_soft_limit, gateway_worker_hard_limit, gateway_max_retries) were
deliberately excluded — grep confirms zero usage of those fields anywhere in
this codebase and zero mentions in architecture/Reference_HLD.md. They belong
to the separate llm_gateway service, not this token manager's resiliency
layer. Rate limiting is a distinct concern and gets its own domain file later.
"""

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class ResiliencySettings(BaseSettings):
    """Circuit breaker and backpressure configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Circuit breaker — three independent breakers: DB, Redis, RabbitMQ.
    # State is stored in Redis so all FastAPI replicas share the same
    # open/closed view.
    cb_db_failure_threshold: int = Field(
        ..., description="DB circuit breaker: consecutive failures before OPEN"
    )
    cb_db_recovery_timeout: int = Field(
        ..., description="DB circuit breaker: seconds in OPEN before HALF_OPEN probe"
    )
    cb_redis_failure_threshold: int = Field(
        ..., description="Redis circuit breaker: consecutive failures before OPEN"
    )
    cb_redis_recovery_timeout: int = Field(
        ...,
        description="Redis circuit breaker: seconds in OPEN before HALF_OPEN probe",
    )
    cb_rmq_failure_threshold: int = Field(
        ..., description="RabbitMQ circuit breaker: consecutive failures before OPEN"
    )
    cb_rmq_recovery_timeout: int = Field(
        ...,
        description="RabbitMQ circuit breaker: seconds in OPEN before HALF_OPEN probe",
    )

    # Backpressure — fail-fast 503 when the system is demonstrably saturated.
    bp_max_queue_depth: int = Field(
        ..., description="Max RabbitMQ token allocation queue depth before 503"
    )
    bp_db_pool_saturation_pct: int = Field(
        ...,
        description="DB pool utilization % (0-100) above which back pressure kicks in",
    )
    bp_drain_rate_per_second: int = Field(
        ...,
        description="Assumed queue drain rate used to estimate Retry-After values",
    )
    bp_retry_after_cap_seconds: int = Field(
        ..., description="Upper bound for Retry-After returned during backpressure"
    )
    bp_queue_safe_depth_ratio: float = Field(
        ...,
        description="Queue depth ratio below which the system is considered healthy",
    )
    bp_db_pool_retry_after_seconds: int = Field(
        ..., description="Retry-After returned when the DB pool is saturated"
    )
    bp_queue_depth_publish_interval_secs: int = Field(
        ...,
        description="Seconds between RabbitMQ work-queue depth publications to Redis",
    )

    # Reconciliation — corrects Redis/PostgreSQL token counter drift.
    # Named for the concern, not the execution vehicle: the old celery_ prefix
    # named the scheduler that happened to invoke these, not what they mean.
    reconcile_interval_secs: int = Field(
        ...,
        description=(
            "Seconds between Redis and PostgreSQL token counter reconciliation runs"
        ),
    )
    reconcile_drift_warning_threshold: int = Field(
        ...,
        description=(
            "Per-deployment token drift magnitude that triggers a warning log "
            "during reconciliation"
        ),
    )
    cleanup_interval_secs: int = Field(
        ...,
        description=("Seconds between expired-allocation cleanup sweeps in PostgreSQL"),
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
                settings_cls, yaml_file=CONFIG_YAML_DIR / "resiliency.yaml"
            ),
            file_secret_settings,
        )

    @field_validator(
        "bp_max_queue_depth",
        "bp_drain_rate_per_second",
        "bp_retry_after_cap_seconds",
        "bp_db_pool_retry_after_seconds",
        "bp_queue_depth_publish_interval_secs",
        "reconcile_interval_secs",
        "reconcile_drift_warning_threshold",
        "cleanup_interval_secs",
    )
    @classmethod
    def validate_positive_backpressure_integers(cls, v: int) -> int:
        """Validate that these backpressure settings are positive."""
        if v <= 0:
            raise ValueError("Setting must be greater than 0")
        return v

    @field_validator("bp_db_pool_saturation_pct")
    @classmethod
    def validate_backpressure_saturation_percent(cls, v: int) -> int:
        """Validate DB pool saturation threshold is a real percentage."""
        if not 1 <= v <= 100:
            raise ValueError("bp_db_pool_saturation_pct must be between 1 and 100")
        return v

    @field_validator("bp_queue_safe_depth_ratio")
    @classmethod
    def validate_backpressure_safe_depth_ratio(cls, v: float) -> float:
        """Validate queue safe-depth ratio remains within (0, 1]."""
        if not 0 < v <= 1:
            raise ValueError(
                "bp_queue_safe_depth_ratio must be greater than 0 and at most 1"
            )
        return v
