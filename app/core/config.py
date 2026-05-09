"""
Application settings and runtime configuration.

Architecture:
-------------
    ┌─────────────────────────────┐
    │ FastAPI / Celery bootstrap  │
    └──────────────┬──────────────┘
                   │ reads
                   ▼
    ┌─────────────────────────────┐
    │ ApplicationSettings         │
    │ (app/core/config.py)        │
    └───────┬───────────┬─────────┘
            │           │
            ▼           ▼
    ┌──────────────┐  ┌─────────────────┐
    │ core/        │  │ resilience/     │
    │ database     │  │ queue/worker    │
    └──────────────┘  └─────────────────┘

This module is the single source of truth for environment-driven application
configuration. It uses Pydantic Settings to load, validate, and expose typed
runtime settings for infrastructure, API, authentication, and async workers.
"""

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Main application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application metadata
    app_name: str = Field(default="LLM Token Manager", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_environment: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # FastAPI server configuration
    fastapi_host: str = Field(default="localhost", description="FastAPI host")
    fastapi_port: int = Field(default=8000, description="FastAPI port")

    # PostgreSQL database configuration
    database_host: str = Field(default="localhost", description="PostgreSQL host")
    database_port: int = Field(default=5432, description="PostgreSQL port")
    database_user: str = Field(default="myuser", description="PostgreSQL user")
    database_password: str = Field(
        default="mypassword", description="PostgreSQL password"
    )
    database_name: str = Field(default="mydb", description="PostgreSQL database name")
    database_pool_size: int = Field(default=20, description="Connection pool size")
    database_max_overflow: int = Field(
        default=10, description="Max overflow connections"
    )

    # Redis configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_max_connections: int = Field(default=50, description="Redis max connections")

    # RabbitMQ configuration
    rabbitmq_host: str = Field(default="localhost", description="RabbitMQ host")
    rabbitmq_port: int = Field(default=5672, description="RabbitMQ port")
    rabbitmq_user: str = Field(default="rmq_user", description="RabbitMQ user")
    rabbitmq_password: str = Field(
        default="rmq_password", description="RabbitMQ password"
    )
    rabbitmq_vhost: str = Field(default="/", description="RabbitMQ virtual host")

    # Celery configuration
    celery_broker_url: str | None = Field(default=None, description="Celery broker URL")
    celery_result_backend: str = Field(
        default="rpc://", description="Celery result backend"
    )
    celery_worker_concurrency: int = Field(
        default=10, description="Celery worker concurrency"
    )
    celery_task_soft_time_limit: int = Field(
        default=300, description="Task soft time limit (seconds)"
    )
    celery_task_time_limit: int = Field(
        default=600, description="Task hard time limit (seconds)"
    )
    require_celery_worker_on_startup: bool = Field(
        default=False,
        description=(
            "Require a healthy Celery worker during FastAPI startup. "
            "Enterprise default is false to keep API and workers "
            "deployable independently."
        ),
    )

    # Rate limiting configuration
    rate_limit_requests_per_minute: int = Field(
        default=100, description="Max requests per minute per user"
    )
    rate_limit_window_seconds: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Endpoint-specific rate limits (enterprise defaults)
    rate_limit_login_per_minute: int = Field(
        default=10, description="Max login attempts per minute per IP+username"
    )
    rate_limit_token_generate_per_minute: int = Field(
        default=30, description="Max token generate requests per minute per IP"
    )
    rate_limit_token_refresh_per_minute: int = Field(
        default=60, description="Max token refresh requests per minute per IP"
    )

    # Per-service rate limiting (token allocation endpoints)
    # X-Service-Id header buckets each upstream microservice independently.
    rate_limit_token_acquire_per_minute: int = Field(
        default=500,
        description="Max /acquire requests per minute per service × IP pair",
    )

    # -------------------------------------------------------------------------
    # PgBouncer connection pooler
    # -------------------------------------------------------------------------
    # The app connects to PgBouncer (port 6432) instead of PostgreSQL directly.
    # PgBouncer multiplexes those connections onto a smaller PostgreSQL pool
    # (transaction mode), which is critical for high-write async workloads.
    #
    # Laptop default: pgbouncer runs in docker-compose on port 6432.
    # Production: point at your PgBouncer cluster / managed pooler
    # (e.g. Supabase Pooler).
    pgbouncer_enabled: bool = Field(
        default=True,
        description="Route DB connections through PgBouncer when True",
    )
    pgbouncer_host: str = Field(default="localhost", description="PgBouncer host")
    pgbouncer_port: int = Field(default=6432, description="PgBouncer port")

    # -------------------------------------------------------------------------
    # Circuit Breaker settings
    # -------------------------------------------------------------------------
    # Three independent breakers: DB, Redis, RabbitMQ.
    # State is stored in Redis so all FastAPI replicas share the same open/closed view.
    #
    # failure_threshold  — consecutive failures to trip OPEN
    # recovery_timeout   — seconds to wait before attempting HALF_OPEN probe
    cb_db_failure_threshold: int = Field(
        default=5,
        description="DB circuit breaker: consecutive failures before OPEN",
    )
    cb_db_recovery_timeout: int = Field(
        default=30,
        description="DB circuit breaker: seconds in OPEN before HALF_OPEN probe",
    )
    cb_redis_failure_threshold: int = Field(
        default=3,
        description="Redis circuit breaker: consecutive failures before OPEN",
    )
    cb_redis_recovery_timeout: int = Field(
        default=10,
        description="Redis circuit breaker: seconds in OPEN before HALF_OPEN probe",
    )
    cb_rmq_failure_threshold: int = Field(
        default=3,
        description="RabbitMQ circuit breaker: consecutive failures before OPEN",
    )
    cb_rmq_recovery_timeout: int = Field(
        default=15,
        description="RabbitMQ circuit breaker: seconds in OPEN before HALF_OPEN probe",
    )

    # -------------------------------------------------------------------------
    # Back Pressure settings
    # -------------------------------------------------------------------------
    # Fail-fast 503 when the system is demonstrably saturated.
    # This protects downstream services and prevents thundering herd cascades.
    #
    # bp_max_queue_depth       — max pending messages in token allocation queue
    # bp_db_pool_saturation_pct — reject when DB pool checked-out ratio exceeds this
    bp_max_queue_depth: int = Field(
        default=10_000,
        description="Max RabbitMQ token allocation queue depth before 503",
    )
    bp_db_pool_saturation_pct: int = Field(
        default=90,
        description="DB pool utilization % (0-100) above which back pressure kicks in",
    )
    bp_drain_rate_per_second: int = Field(
        default=400,
        description="Assumed queue drain rate used to estimate Retry-After values",
    )
    bp_retry_after_cap_seconds: int = Field(
        default=60,
        description="Upper bound for Retry-After returned during backpressure",
    )
    bp_queue_safe_depth_ratio: float = Field(
        default=0.8,
        description="Queue depth ratio below which the system is considered healthy",
    )
    bp_db_pool_retry_after_seconds: int = Field(
        default=5,
        description="Retry-After returned when the DB pool is saturated",
    )
    bp_queue_depth_publish_interval_secs: int = Field(
        default=5,
        description="Seconds between RabbitMQ work-queue depth publications to Redis",
    )

    # -------------------------------------------------------------------------
    # Redis Token Counter (fast path)
    # -------------------------------------------------------------------------
    # Atomic Lua-based token counters eliminate DB reads on the acquire hot path.
    # Keys are auto-seeded at startup from PostgreSQL and periodically reconciled.
    redis_token_counter_ttl_secs: int = Field(
        default=3_600,
        description=(
            "TTL for Redis token counter keys (seconds). Reconciler re-seeds on expiry."
        ),
    )
    redis_token_counter_db: int = Field(
        default=1,
        description=(
            "Redis logical DB index for token counters "
            "(isolates from rate limiter DB 0)"
        ),
    )
    redis_token_counter_max_connections: int = Field(
        default=20,
        description="Dedicated Redis connection limit for token counter operations",
    )

    # -------------------------------------------------------------------------
    # RabbitMQ token allocation queue topology
    # -------------------------------------------------------------------------
    rabbitmq_token_exchange_name: str = Field(
        default="token.allocation",
        description="Primary RabbitMQ exchange name for token allocation messages",
    )
    rabbitmq_token_exchange_type: str = Field(
        default="direct",
        description="RabbitMQ exchange type used for token allocation messages",
    )
    rabbitmq_token_dlx_name: str = Field(
        default="token.allocation.dlx",
        description="Dead-letter exchange name for token allocation failures",
    )
    rabbitmq_token_work_queue_name: str = Field(
        default="token.allocation.work",
        description="Primary RabbitMQ work queue for token allocation persistence",
    )
    rabbitmq_token_dlq_queue_name: str = Field(
        default="token.allocation.dlq",
        description="RabbitMQ dead-letter queue for failed token allocation messages",
    )
    rabbitmq_token_allocate_routing_key: str = Field(
        default="token.allocate",
        description="Routing key for token allocation persistence messages",
    )
    rabbitmq_token_allocate_dead_routing_key: str = Field(
        default="token.allocate.dead",
        description="Routing key for token allocation dead-letter messages",
    )
    rabbitmq_token_queue_message_ttl_ms: int = Field(
        default=300_000,
        description="TTL in milliseconds for token allocation work queue messages",
    )
    rabbitmq_token_queue_delivery_limit: int = Field(
        default=6,
        description=(
            "Maximum RabbitMQ quorum redeliveries before dead-lettering. "
            "Must exceed the configured retry stage count so the consumer can "
            "perform explicit DLQ routing after the final retry."
        ),
    )
    rabbitmq_token_heartbeat_seconds: int = Field(
        default=60,
        description="Heartbeat interval in seconds for token queue broker connections",
    )
    token_queue_connection_pool_limit: int = Field(
        default=10,
        description="Persistent Kombu connection-pool size for Layer 4 publishing",
    )
    token_queue_retry_schedule_seconds: tuple[int, ...] = Field(
        default=(5, 10, 20, 40, 60),
        description=(
            "Retry schedule in seconds for token allocation persistence messages "
            "before DLQ routing"
        ),
    )
    token_queue_consumer_prefetch_count: int = Field(
        default=20,
        description=(
            "Number of unacknowledged work messages each token queue consumer "
            "process may hold at once"
        ),
    )
    token_queue_consumer_concurrency: int = Field(
        default=8,
        description=(
            "Number of raw token queue consumer processes to run for Layer 4 "
            "persistence throughput"
        ),
    )
    token_queue_consumer_requeue_backoff_seconds: int = Field(
        default=1,
        description=(
            "Seconds to pause before requeueing a work message when retry or DLQ "
            "publishing is temporarily unavailable"
        ),
    )

    # -------------------------------------------------------------------------
    # Celery Beat / Worker periodic task intervals
    # -------------------------------------------------------------------------
    celery_token_maintenance_queue_name: str = Field(
        default="token.maintenance",
        description="Dedicated Celery queue for token maintenance tasks",
    )
    celery_reconcile_interval_secs: int = Field(
        default=60,
        description=(
            "Seconds between Redis and PostgreSQL token counter reconciliation runs"
        ),
    )
    celery_cleanup_interval_secs: int = Field(
        default=300,
        description="Seconds between expired allocation cleanup runs",
    )
    celery_dlq_alert_threshold: int = Field(
        default=100,
        description="DLQ message count that triggers an alert log",
    )
    celery_token_persist_max_retries: int = Field(
        default=3,
        description=(
            "Deprecated compatibility setting for Celery token persistence retries. "
            "Layer 4 queue retries are now controlled by "
            "token_queue_retry_schedule_seconds."
        ),
    )
    celery_token_persist_retry_base_seconds: int = Field(
        default=5,
        description=(
            "Deprecated compatibility setting for Celery token persistence retry base "
            "delay. Layer 4 queue retries are now controlled by "
            "token_queue_retry_schedule_seconds."
        ),
    )
    celery_token_persist_retry_backoff_multiplier: int = Field(
        default=5,
        description=(
            "Deprecated compatibility setting for Celery token persistence retry "
            "backoff. Layer 4 queue retries are now controlled by "
            "token_queue_retry_schedule_seconds."
        ),
    )
    celery_token_task_soft_time_limit_seconds: int = Field(
        default=20,
        description="Soft time limit for token allocation Celery tasks",
    )
    celery_token_task_time_limit_seconds: int = Field(
        default=30,
        description="Hard time limit for token allocation Celery tasks",
    )

    # Caching configuration
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")

    # LLM Provider API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    azure_openai_api_key: str | None = Field(
        default=None, description="Azure OpenAI API key"
    )
    azure_openai_endpoint: str | None = Field(
        default=None, description="Azure OpenAI endpoint"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    google_api_key: str | None = Field(default=None, description="Google API key")

    # Default LLM settings
    default_max_tokens: int = Field(default=1000, description="Default max tokens")
    default_temperature: float = Field(default=0.7, description="Default temperature")

    # JWT Authentication configuration
    jwt_secret_key: str = Field(
        default="CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET",
        description="JWT secret key for token signing",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_hours: int = Field(
        default=24, description="Access token expiration in hours"
    )
    jwt_refresh_enabled: bool = Field(
        default=False, description="Enable refresh token support"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration in days"
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

    @field_validator("default_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator(
        "bp_max_queue_depth",
        "bp_drain_rate_per_second",
        "bp_retry_after_cap_seconds",
        "bp_db_pool_retry_after_seconds",
        "bp_queue_depth_publish_interval_secs",
        "redis_token_counter_ttl_secs",
        "redis_token_counter_max_connections",
        "rabbitmq_token_queue_message_ttl_ms",
        "rabbitmq_token_queue_delivery_limit",
        "rabbitmq_token_heartbeat_seconds",
        "token_queue_connection_pool_limit",
        "token_queue_consumer_prefetch_count",
        "token_queue_consumer_concurrency",
        "token_queue_consumer_requeue_backoff_seconds",
        "celery_reconcile_interval_secs",
        "celery_cleanup_interval_secs",
        "celery_dlq_alert_threshold",
        "celery_token_persist_max_retries",
        "celery_token_persist_retry_base_seconds",
        "celery_token_persist_retry_backoff_multiplier",
        "celery_token_task_soft_time_limit_seconds",
        "celery_token_task_time_limit_seconds",
    )
    @classmethod
    def validate_positive_resilience_integers(cls, v: int) -> int:
        """Validate that resilience integer settings are positive."""
        if v <= 0:
            raise ValueError("Resilience setting must be greater than 0")
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

    @field_validator(
        "rabbitmq_token_exchange_name",
        "rabbitmq_token_exchange_type",
        "rabbitmq_token_dlx_name",
        "rabbitmq_token_work_queue_name",
        "rabbitmq_token_dlq_queue_name",
        "rabbitmq_token_allocate_routing_key",
        "rabbitmq_token_allocate_dead_routing_key",
        "celery_token_maintenance_queue_name",
    )
    @classmethod
    def validate_non_empty_resilience_names(cls, v: str) -> str:
        """Validate RabbitMQ resilience names are not blank."""
        normalized = v.strip()
        if not normalized:
            raise ValueError("Resilience queue and routing names must not be blank")
        return normalized

    @field_validator("token_queue_retry_schedule_seconds", mode="before")
    @classmethod
    def validate_token_queue_retry_schedule(
        cls, value: object
    ) -> tuple[int, ...] | object:
        """Normalize retry schedules from tuple/list/csv env forms."""
        if isinstance(value, str):
            parsed = tuple(
                int(part.strip()) for part in value.split(",") if part.strip()
            )
        elif isinstance(value, list):
            parsed = tuple(int(part) for part in value)
        else:
            parsed = value

        if not isinstance(parsed, tuple) or not parsed:
            raise ValueError(
                "token_queue_retry_schedule_seconds must contain at least one retry"
            )
        if any(delay <= 0 for delay in parsed):
            raise ValueError(
                "token_queue_retry_schedule_seconds values must be greater than 0"
            )
        if tuple(sorted(parsed)) != parsed:
            raise ValueError(
                "token_queue_retry_schedule_seconds must be in ascending order"
            )
        return parsed

    @model_validator(mode="after")
    def validate_resilience_time_limits(self) -> "ApplicationSettings":
        """Validate related resilience settings as a coherent set."""
        if (
            self.celery_token_task_soft_time_limit_seconds
            >= self.celery_token_task_time_limit_seconds
        ):
            raise ValueError(
                "celery_token_task_soft_time_limit_seconds must be less than "
                "celery_token_task_time_limit_seconds"
            )
        minimum_delivery_limit = len(self.token_queue_retry_schedule_seconds) + 1
        if self.rabbitmq_token_queue_delivery_limit < minimum_delivery_limit:
            raise ValueError(
                "rabbitmq_token_queue_delivery_limit must be at least "
                f"{minimum_delivery_limit} so the final retry can reach the "
                "consumer for explicit DLQ routing"
            )
        return self

    @property
    def database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        return (
            f"postgresql+asyncpg://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def database_url_sync(self) -> str:
        """Construct sync PostgreSQL database URL (for postgres_schema)."""
        return (
            f"postgresql://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis URL (DB 0 — rate limiting / general cache)."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def redis_token_counter_url(self) -> str:
        """Construct Redis URL for the token counter fast path on isolated DB 1."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
                f"/{self.redis_token_counter_db}"
            )
        return (
            f"redis://{self.redis_host}:{self.redis_port}/{self.redis_token_counter_db}"
        )

    @property
    def broker_url(self) -> str:
        """Construct RabbitMQ broker URL."""
        if self.celery_broker_url:
            return self.celery_broker_url
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"
        )

    @property
    def effective_database_url(self) -> str:
        """
        Return the database URL the application should use.

        When pgbouncer_enabled=True, routes through PgBouncer (transaction-mode pooler)
        instead of connecting directly to PostgreSQL.

        PgBouncer benefits for high-write workloads:
        - Multiplexes hundreds of app connections onto a small PostgreSQL pool
        - Reduces PostgreSQL memory pressure (each connection ~10MB)
        - Absorbs connection storms during traffic bursts

        Production note: when routing through PgBouncer in transaction mode,
        set SQLAlchemy pool_size=1 and use NullPool to avoid double-pooling.
        """
        if self.pgbouncer_enabled:
            return (
                f"postgresql+asyncpg://{self.database_user}:{self.database_password}"
                f"@{self.pgbouncer_host}:{self.pgbouncer_port}/{self.database_name}"
            )
        return self.database_url


# Global runtime settings singleton
settings = ApplicationSettings()
