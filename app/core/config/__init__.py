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

from __future__ import annotations

from pydantic import model_validator
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

# Fixture defaults such as "mypassword" only exist so ApplicationSettings()
# can construct in dev/test without a .env file. Recognising them as a
# canonical set (not a length/entropy heuristic) means
# validate_production_infrastructure_secrets below can never be satisfied by
# accident — only by literally shipping the sample value to production.
_KNOWN_INSECURE_DEFAULTS = frozenset({"mypassword", "rmq_password"})


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

    @model_validator(mode="after")
    def validate_jwt_configuration(self) -> ApplicationSettings:
        """Reject unsafe signing configuration before serving production traffic."""
        if self.jwt_algorithm not in {"HS256", "HS384", "HS512"}:
            raise ValueError("JWT_ALGORITHM must be a supported HMAC algorithm")
        jwt_secret = self.jwt_secret_key.get_secret_value()
        if self.app_environment == "production" and (
            jwt_secret == "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET"
            or len(jwt_secret.encode("utf-8")) < 32
        ):
            raise ValueError(
                "Production requires a unique JWT secret of at least 32 bytes"
            )
        return self

    @model_validator(mode="after")
    def validate_production_infrastructure_secrets(self) -> ApplicationSettings:
        """
        Fail closed if PostgreSQL or RabbitMQ still use their sample .env passwords.

        Mirrors validate_jwt_configuration's intent for the other two
        credentialed dependencies: a sample password reaching production is a
        configuration bug, not a valid (if weak) operating point, so this
        raises at startup — the same place a missing required field would —
        rather than letting the process boot with a guessable database or
        broker password.
        """
        if self.app_environment != "production":
            return self
        insecure_fields = {
            "database_password": self.database_password,
            "rabbitmq_password": self.rabbitmq_password,
        }
        for field_name, secret in insecure_fields.items():
            if secret.get_secret_value() in _KNOWN_INSECURE_DEFAULTS:
                raise ValueError(
                    f"Production requires a unique {field_name}, not the "
                    "sample .env default"
                )
        return self

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
