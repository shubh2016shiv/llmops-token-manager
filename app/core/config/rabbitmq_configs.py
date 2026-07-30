"""
RabbitMQ connection and token allocation queue topology settings.

Defaults live in yaml/rabbitmq.yaml and are directly editable there.
rabbitmq_password and celery_broker_url are secret/secret-adjacent and are
intentionally absent from the YAML defaults — they are sourced from .env
only. Values set in .env (or the process environment) override the YAML
defaults for every other field.

Scope note: Celery worker/task tuning (celery_result_backend,
celery_worker_concurrency, celery_task_soft_time_limit, etc.) is a distinct
business concern — worker/task behavior, not broker connection or queue
topology — and gets its own domain file.
"""

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class RabbitMQSettings(BaseSettings):
    """RabbitMQ connection and token allocation queue topology configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    rabbitmq_host: str = Field(..., description="RabbitMQ host")
    rabbitmq_port: int = Field(..., description="RabbitMQ port")
    rabbitmq_user: str = Field(..., description="RabbitMQ user")
    rabbitmq_password: str = Field(
        default="rmq_password",
        description="RabbitMQ password (secret — set via .env only)",
    )
    rabbitmq_vhost: str = Field(..., description="RabbitMQ virtual host")

    celery_broker_url: str | None = Field(
        default=None,
        description=(
            "Full Celery/AMQP broker URL override (secret-adjacent — set via "
            ".env only). When set, takes precedence over the host/user/"
            "password/vhost composition in broker_url."
        ),
    )

    # RabbitMQ token allocation queue topology
    rabbitmq_token_exchange_name: str = Field(
        ..., description="Primary RabbitMQ exchange name for token allocation messages"
    )
    rabbitmq_token_exchange_type: str = Field(
        ..., description="RabbitMQ exchange type used for token allocation messages"
    )
    rabbitmq_token_dlx_name: str = Field(
        ..., description="Dead-letter exchange name for token allocation failures"
    )
    rabbitmq_token_work_queue_name: str = Field(
        ..., description="Primary RabbitMQ work queue for token allocation persistence"
    )
    rabbitmq_token_dlq_queue_name: str = Field(
        ...,
        description="RabbitMQ dead-letter queue for failed token allocation messages",
    )
    rabbitmq_token_allocate_routing_key: str = Field(
        ..., description="Routing key for token allocation persistence messages"
    )
    rabbitmq_token_allocate_dead_routing_key: str = Field(
        ..., description="Routing key for token allocation dead-letter messages"
    )
    rabbitmq_token_queue_message_ttl_ms: int = Field(
        ..., description="TTL in milliseconds for token allocation work queue messages"
    )
    rabbitmq_token_queue_delivery_limit: int = Field(
        ...,
        description=(
            "Maximum RabbitMQ quorum redeliveries before dead-lettering. "
            "Must exceed the configured retry stage count so the consumer can "
            "perform explicit DLQ routing after the final retry."
        ),
    )
    rabbitmq_token_heartbeat_seconds: int = Field(
        ...,
        description="Heartbeat interval in seconds for token queue broker connections",
    )
    token_queue_connection_pool_limit: int = Field(
        ..., description="Persistent Kombu connection-pool size for Layer 4 publishing"
    )
    token_queue_retry_schedule_seconds: tuple[int, ...] = Field(
        ...,
        description=(
            "Retry schedule in seconds for token allocation persistence messages "
            "before DLQ routing"
        ),
    )
    token_queue_consumer_prefetch_count: int = Field(
        ...,
        description=(
            "Number of unacknowledged work messages each token queue consumer "
            "process may hold at once"
        ),
    )
    token_queue_consumer_concurrency: int = Field(
        ...,
        description=(
            "Number of raw token queue consumer processes to run for Layer 4 "
            "persistence throughput"
        ),
    )
    token_queue_consumer_requeue_backoff_seconds: int = Field(
        ...,
        description=(
            "Seconds to pause before requeueing a work message when retry or DLQ "
            "publishing is temporarily unavailable"
        ),
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
                settings_cls, yaml_file=CONFIG_YAML_DIR / "rabbitmq.yaml"
            ),
            file_secret_settings,
        )

    @field_validator(
        "rabbitmq_token_exchange_name",
        "rabbitmq_token_exchange_type",
        "rabbitmq_token_dlx_name",
        "rabbitmq_token_work_queue_name",
        "rabbitmq_token_dlq_queue_name",
        "rabbitmq_token_allocate_routing_key",
        "rabbitmq_token_allocate_dead_routing_key",
    )
    @classmethod
    def validate_non_empty_rabbitmq_names(cls, v: str) -> str:
        """Validate RabbitMQ topology names are not blank."""
        normalized = v.strip()
        if not normalized:
            raise ValueError("RabbitMQ queue and routing names must not be blank")
        return normalized

    @field_validator(
        "rabbitmq_token_queue_message_ttl_ms",
        "rabbitmq_token_queue_delivery_limit",
        "rabbitmq_token_heartbeat_seconds",
        "token_queue_connection_pool_limit",
        "token_queue_consumer_prefetch_count",
        "token_queue_consumer_concurrency",
        "token_queue_consumer_requeue_backoff_seconds",
    )
    @classmethod
    def validate_positive_rabbitmq_integers(cls, v: int) -> int:
        """Validate that these RabbitMQ settings are positive."""
        if v <= 0:
            raise ValueError("Setting must be greater than 0")
        return v

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
    def validate_delivery_limit_covers_retry_schedule(self) -> "RabbitMQSettings":
        """Ensure the redelivery limit lets the final retry reach the consumer."""
        minimum_delivery_limit = len(self.token_queue_retry_schedule_seconds) + 1
        if self.rabbitmq_token_queue_delivery_limit < minimum_delivery_limit:
            raise ValueError(
                "rabbitmq_token_queue_delivery_limit must be at least "
                f"{minimum_delivery_limit} so the final retry can reach the "
                "consumer for explicit DLQ routing"
            )
        return self

    @property
    def broker_url(self) -> str:
        """Construct RabbitMQ broker URL."""
        if self.celery_broker_url:
            return self.celery_broker_url
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"
        )
