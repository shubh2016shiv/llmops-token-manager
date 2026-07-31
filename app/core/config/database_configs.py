"""
PostgreSQL connection and PgBouncer pooling settings.

Defaults live in yaml/database.yaml and are directly editable there.
database_password is a secret and is intentionally absent from the YAML
defaults — it is sourced from .env only. Values set in .env (or the process
environment) override the YAML defaults for every other field.
"""

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and PgBouncer pooling configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_host: str = Field(..., description="PostgreSQL host")
    database_port: int = Field(..., description="PostgreSQL port")
    database_user: str = Field(..., description="PostgreSQL user")
    database_password: str = Field(
        default="mypassword",
        description="PostgreSQL password (secret — set via .env only)",
    )
    database_name: str = Field(..., description="PostgreSQL database name")
    database_pool_size: int = Field(..., description="Connection pool size")
    database_max_overflow: int = Field(..., description="Max overflow connections")
    database_pool_recycle_seconds: int = Field(
        ...,
        description=(
            "Recycle a pooled SQLAlchemy connection after this many seconds. "
            "Should match PgBouncer's server_lifetime "
            "(infra/config/pgbouncer/pgbouncer.ini) so neither side outlives "
            "the other's assumption about connection freshness."
        ),
    )
    database_pool_timeout_seconds: int = Field(
        ..., description="Seconds to wait for a pooled connection before erroring"
    )

    pgbouncer_enabled: bool = Field(
        ..., description="Route DB connections through PgBouncer when True"
    )
    pgbouncer_host: str = Field(..., description="PgBouncer host")
    pgbouncer_port: int = Field(..., description="PgBouncer port")

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
                settings_cls, yaml_file=CONFIG_YAML_DIR / "database.yaml"
            ),
            file_secret_settings,
        )

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
    def effective_database_url(self) -> str:
        """
        Return the database URL the application should use.

        When pgbouncer_enabled=True, routes through PgBouncer (transaction-mode
        pooler) instead of connecting directly to PostgreSQL.

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
