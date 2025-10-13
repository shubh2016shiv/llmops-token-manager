"""
Configuration Manager
--------------------
Centralized configuration management using Pydantic Settings.
All application settings are loaded from environment variables with validation.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class ApplicationSettings(BaseSettings):
    """Main application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application metadata
    app_name: str = Field(default="LLM Token Manager", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # FastAPI server configuration
    fastapi_host: str = Field(default="0.0.0.0", description="FastAPI host")
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
    redis_password: Optional[str] = Field(default=None, description="Redis password")
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
    celery_broker_url: Optional[str] = Field(
        default=None, description="Celery broker URL"
    )
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

    # Rate limiting configuration
    rate_limit_requests_per_minute: int = Field(
        default=100, description="Max requests per minute per user"
    )
    rate_limit_window_seconds: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Caching configuration
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")

    # LLM Provider API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    azure_openai_api_key: Optional[str] = Field(
        default=None, description="Azure OpenAI API key"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure OpenAI endpoint"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

    # Default LLM settings
    default_max_tokens: int = Field(default=1000, description="Default max tokens")
    default_temperature: float = Field(default=0.7, description="Default temperature")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is acceptable."""
        valid_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("default_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

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
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def broker_url(self) -> str:
        """Construct RabbitMQ broker URL."""
        if self.celery_broker_url:
            return self.celery_broker_url
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"
        )


# Global settings instance
settings = ApplicationSettings()
