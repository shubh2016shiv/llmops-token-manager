"""
JWT authentication settings.

Defaults live in yaml/jwt.yaml and are directly editable there.
jwt_secret_key is a secret and is intentionally absent from the YAML
defaults — it is sourced from .env only. Values set in .env (or the process
environment) override the YAML defaults for every other field.
"""

from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from app.core.config.constants import CONFIG_YAML_DIR


class JWTSettings(BaseSettings):
    """JWT authentication configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # SecretStr (not str): keeps the key out of repr()/str()/logging/model_dump()
    # by accident. Callers that need the raw value for jose.jwt encode/decode
    # call .get_secret_value() explicitly at the point of use — an intentional,
    # greppable seam rather than an implicit string leak.
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET"),
        description="JWT secret key for token signing (secret — set via .env only)",
    )
    jwt_algorithm: str = Field(..., description="JWT algorithm")
    jwt_access_token_expire_hours: int = Field(
        ..., description="Access token expiration in hours"
    )
    jwt_refresh_enabled: bool = Field(..., description="Enable refresh token support")
    jwt_refresh_token_expire_days: int = Field(
        ..., description="Refresh token expiration in days"
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
                settings_cls, yaml_file=CONFIG_YAML_DIR / "jwt.yaml"
            ),
            file_secret_settings,
        )
