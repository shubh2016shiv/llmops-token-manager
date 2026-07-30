"""
Application settings - the composed public surface of app.core.config.

Architecture:
-------------
    ┌─────────────────────────────────────────────┐
    │ FastAPI / workers: from app.core.config     │
    │                    import settings          │
    └───────────────────────┬─────────────────────┘
                            │
    ┌───────────────────────▼─────────────────────┐
    │ ApplicationSettings (this module)           │
    │ composes the per-domain settings classes    │
    └───────────────────────┬─────────────────────┘
                            │
    ┌───────────────────────▼─────────────────────┐
    │ app_configs / database_configs /            │
    │ redis_configs / rabbitmq_configs /          │
    │ resiliency_configs / rate_limit_configs /   │
    │ jwt_configs   (fields, validators, URLs)    │
    └─────────────────────────────────────────────┘

Each domain module owns its own fields, validators, and computed URLs, with
editable defaults in yaml/<domain>.yaml. This module owns only composition and
exposure: it defines no fields and no business logic of its own.

Resolution order (highest wins): init args > environment variables > .env >
yaml/<domain>.yaml > file secrets. Reserve .env for secrets and
per-environment overrides; edit the YAML files for baseline behavior.
"""

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.app_configs import AppSettings
from app.core.config.constants import CONFIG_YAML_DIR
from app.core.config.database_configs import DatabaseSettings
from app.core.config.jwt_configs import JWTSettings
from app.core.config.rabbitmq_configs import RabbitMQSettings
from app.core.config.rate_limit_configs import RateLimitSettings
from app.core.config.redis_configs import RedisSettings
from app.core.config.resiliency_configs import ResiliencySettings

DOMAIN_YAML_FILENAMES = (
    "app.yaml",
    "database.yaml",
    "redis.yaml",
    "rabbitmq.yaml",
    "resiliency.yaml",
    "rate_limit.yaml",
    "jwt.yaml",
)


class ApplicationSettings(
    AppSettings,
    DatabaseSettings,
    RedisSettings,
    RabbitMQSettings,
    ResiliencySettings,
    RateLimitSettings,
    JWTSettings,
):
    """
    Every application setting, composed from the per-domain settings classes.

    Fields, validators, and computed URL properties are inherited unchanged
    from the domain classes. Nothing is declared here: to add or change a
    setting, edit the owning domain module and its YAML defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
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
        """
        Load every domain's YAML defaults beneath the environment sources.

        This override is required, not boilerplate: each domain class defines
        its own settings_customise_sources, and Python's MRO would silently
        keep only the first one - loading a single domain's YAML and leaving
        every other domain's fields unresolved.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=[
                    CONFIG_YAML_DIR / filename for filename in DOMAIN_YAML_FILENAMES
                ],
            ),
            file_secret_settings,
        )


# Global runtime settings singleton
settings = ApplicationSettings()

__all__ = ["ApplicationSettings", "settings"]
