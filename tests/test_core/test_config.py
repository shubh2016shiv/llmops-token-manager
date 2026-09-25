"""
Unit tests for app.core.config.

Scope note: this test file previously asserted fields (celery_result_backend,
openai_api_key, anthropic_api_key, default_temperature, ...) that belonged to
the pre-decoupling monolithic config (app/core/config_OBSELETE.py). Those
concerns moved to llm_services; ApplicationSettings here composes only the
seven domain classes actually listed in app/core/config/__init__.py
(AppSettings, DatabaseSettings, RedisSettings, RabbitMQSettings,
ResiliencySettings, RateLimitSettings, JWTSettings). This rewrite tests that
current, composed surface.

Isolation: this repo's real .env (a developer's local secrets file, not
committed) and the process environment both outrank YAML in
settings_customise_sources. Asserting against literal defaults would either
be flaky (depends on whoever's machine runs the suite) or silently pass
because a real secret happens to match — and it gets worse than "whoever's
.env": `litellm` (imported transitively by app.app, which
tests/test_core/test_app_lifespan.py imports) calls python-dotenv's
`load_dotenv()` at import time, which — unlike pydantic-settings' own
non-mutating dotenv reader — permanently copies the on-disk .env into this
*process's* os.environ. Once that has run, every later test in the same
pytest process sees real .env values as if they were process env vars, no
matter what tests/conftest.py set up front. `isolated_settings` below
defends against both sources at once: it clears every known
ApplicationSettings field's env var name (derived from the model itself, so
it can't drift out of sync with the schema) and builds with `_env_file=None`
(skips .env entirely), so a "defaults" test resolves purely from
yaml/<domain>.yaml — the actual baseline this module promises to load,
regardless of import order or what other tests ran first.
"""

from collections.abc import Iterator

from pydantic import SecretStr, ValidationError
import pytest

from app.core.config import ApplicationSettings


@pytest.fixture
def isolated_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[type[ApplicationSettings]]:
    """
    Yield a factory that builds ApplicationSettings from YAML defaults only.

    Use this (instead of `ApplicationSettings(...)` directly) whenever a test
    asserts a *default* value rather than exercising validation of an
    explicitly-passed value — otherwise the assertion is really testing
    whoever's .env (or whichever earlier test polluted os.environ) happens
    to be on disk.
    """
    for field_name in ApplicationSettings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    def _build(**overrides: object) -> ApplicationSettings:
        return ApplicationSettings(_env_file=None, **overrides)  # type: ignore[call-arg]

    yield _build


class TestApplicationSettingsDefaults:
    """Defaults come from yaml/<domain>.yaml, not from any developer's .env."""

    def test_app_identity_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.app_name == "LLM Token Manager"
        assert settings.app_environment == "development"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.fastapi_host == "localhost"
        assert settings.fastapi_port == 8000

    def test_database_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.database_host == "localhost"
        assert settings.database_port == 5432
        assert settings.database_user == "llm_user"
        assert settings.database_name == "llm_services"
        assert settings.database_pool_size == 20
        assert settings.database_max_overflow == 10
        assert settings.pgbouncer_enabled is True

    def test_redis_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.redis_host == "localhost"
        assert settings.redis_db == 1
        assert settings.redis_password is None
        assert settings.redis_max_connections == 50
        assert settings.cache_enabled is True

    def test_rabbitmq_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.rabbitmq_host == "localhost"
        assert settings.rabbitmq_user == "rmq_user"
        assert settings.rabbitmq_vhost == "/"
        assert settings.celery_broker_url is None
        assert settings.rabbitmq_token_exchange_name == "token.allocation"
        assert settings.token_queue_retry_schedule_seconds == (5, 10, 20, 40, 60)

    def test_resiliency_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.cb_db_failure_threshold == 5
        assert settings.cb_db_recovery_timeout == 30
        assert settings.bp_max_queue_depth == 10000
        assert settings.bp_queue_safe_depth_ratio == 0.8

    def test_rate_limit_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.rate_limit_requests_per_minute == 100
        assert settings.rate_limit_window_seconds == 60
        assert settings.rate_limit_trusted_proxy_hops == 0
        assert settings.rate_limit_trusted_proxy_networks == []

    def test_jwt_defaults(self, isolated_settings):
        settings = isolated_settings()

        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_access_token_expire_hours == 24
        assert settings.jwt_refresh_enabled is False
        # The sample secret is a deliberate, checked-in placeholder — never a
        # real credential — so asserting its value is not a secret leak.
        assert (
            settings.jwt_secret_key.get_secret_value()
            == "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET"
        )


class TestSecretRedaction:
    """
    Secret fields must never surface in repr(), str(), or a plain dump.

    This is the regression test for the SecretStr migration: before it,
    `str(settings)` (as FastAPI's default logging/exception formatting would
    produce) printed the raw JWT/DB/RabbitMQ/Redis passwords in full.
    """

    def test_repr_and_str_never_contain_raw_secret_values(self):
        settings = ApplicationSettings(
            jwt_secret_key="super-secret-value-not-for-logs",
            database_password="super-secret-db-password",
            rabbitmq_password="super-secret-broker-password",
            redis_password="super-secret-redis-password",
        )

        dump = f"{settings!r} {settings}"

        assert "super-secret-value-not-for-logs" not in dump
        assert "super-secret-db-password" not in dump
        assert "super-secret-broker-password" not in dump
        assert "super-secret-redis-password" not in dump
        assert "**********" in dump

    def test_model_dump_masks_secrets_by_default(self):
        settings = ApplicationSettings(database_password="another-secret-password")

        dumped = settings.model_dump()

        assert isinstance(dumped["database_password"], SecretStr)
        assert "another-secret-password" not in str(dumped["database_password"])

    def test_model_dump_json_masks_secrets(self):
        settings = ApplicationSettings(rabbitmq_password="json-secret-password")

        dumped_json = settings.model_dump_json()

        assert "json-secret-password" not in dumped_json
        assert "**********" in dumped_json

    def test_get_secret_value_returns_the_real_value(self):
        """The escape hatch exists and works — redaction isn't silent data loss."""
        settings = ApplicationSettings(jwt_secret_key="round-trips-correctly")

        assert settings.jwt_secret_key.get_secret_value() == "round-trips-correctly"


class TestUnknownFieldHandling:
    """
    Documents a deliberate (not accidental) deviation from "reject unknown
    fields": ApplicationSettings composes seven domain classes over one
    shared .env, so each domain must tolerate the other six domains' keys.
    Setting model_config to extra="forbid" here would also start rejecting
    the llm_services-owned keys (OPENAI_API_KEY, CELERY_RESULT_BACKEND, ...)
    still present in .env.example from before the service split — a
    startup-breaking change tracked separately as part of that decoupling,
    not fixed silently in this pass.
    """

    def test_unrecognized_keys_are_ignored_not_rejected(self):
        settings = ApplicationSettings(
            openai_api_key="not-a-real-field-on-this-service-anymore",
            some_totally_made_up_field="ignored",
        )

        assert not hasattr(settings, "openai_api_key")
        assert not hasattr(settings, "some_totally_made_up_field")


class TestApplicationSettingsValidators:
    """Field and cross-field validators."""

    @pytest.mark.parametrize(
        "valid_level",
        ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "debug", "info"],
    )
    def test_validate_log_level_valid(self, valid_level):
        settings = ApplicationSettings(log_level=valid_level)

        assert settings.log_level == valid_level.upper()

    def test_validate_log_level_invalid(self):
        with pytest.raises(ValidationError, match="Log level must be one of"):
            ApplicationSettings(log_level="INVALID")

    @pytest.mark.parametrize("valid_env", ["development", "staging", "production"])
    def test_validate_app_environment_valid(self, valid_env):
        settings = ApplicationSettings(
            app_environment=valid_env,
            jwt_secret_key="x" * 32,
            database_password="not-a-sample-default",
            rabbitmq_password="not-a-sample-default",
        )

        assert settings.app_environment == valid_env

    def test_validate_app_environment_invalid(self):
        with pytest.raises(ValidationError, match="app_environment must be one of"):
            ApplicationSettings(app_environment="testing")

    @pytest.mark.parametrize(
        "field_name",
        [
            "cb_db_failure_threshold",
            "cb_db_recovery_timeout",
            "cb_redis_failure_threshold",
            "cb_redis_recovery_timeout",
            "cb_rmq_failure_threshold",
            "cb_rmq_recovery_timeout",
            "bp_max_queue_depth",
            "bp_drain_rate_per_second",
            "bp_retry_after_cap_seconds",
            "bp_db_pool_retry_after_seconds",
            "bp_queue_depth_publish_interval_secs",
            "reconcile_interval_secs",
            "reconcile_drift_warning_threshold",
            "cleanup_interval_secs",
        ],
    )
    def test_validate_resiliency_settings_invalid(self, field_name):
        """
        Regression test: cb_* circuit-breaker fields previously had no
        validation at all, so cb_db_recovery_timeout=0 (instant HALF_OPEN
        retries against an already-failing dependency) or
        cb_db_failure_threshold=0 (trips on the first call) would pass
        silently and only surface as a production incident.
        """
        with pytest.raises(ValidationError, match="greater than 0"):
            ApplicationSettings(**{field_name: 0})

    @pytest.mark.parametrize(
        "field_name",
        [
            "rate_limit_requests_per_minute",
            "rate_limit_window_seconds",
            "rate_limit_token_generate_per_minute",
            "rate_limit_token_refresh_per_minute",
            "rate_limit_token_acquire_per_minute",
        ],
    )
    def test_validate_rate_limit_settings_invalid(self, field_name):
        """Regression test: these had no validation before this pass."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ApplicationSettings(**{field_name: 0})

    @pytest.mark.parametrize("saturation_pct", [0, 101])
    def test_validate_backpressure_saturation_percent_invalid(self, saturation_pct):
        with pytest.raises(ValidationError):
            ApplicationSettings(bp_db_pool_saturation_pct=saturation_pct)

    @pytest.mark.parametrize("invalid_ratio", [0.0, -0.1, 1.1])
    def test_validate_backpressure_safe_depth_ratio_invalid(self, invalid_ratio):
        with pytest.raises(ValidationError):
            ApplicationSettings(bp_queue_safe_depth_ratio=invalid_ratio)

    @pytest.mark.parametrize("retry_schedule", ["", "10,5", [5, 0, 10]])
    def test_validate_retry_schedule_invalid(self, retry_schedule):
        with pytest.raises(ValidationError):
            ApplicationSettings(token_queue_retry_schedule_seconds=retry_schedule)

    def test_validate_retry_schedule_csv_normalizes_to_tuple(self):
        settings = ApplicationSettings(token_queue_retry_schedule_seconds="3,6,9")

        assert settings.token_queue_retry_schedule_seconds == (3, 6, 9)

    def test_validate_delivery_limit_exceeds_retry_stage_count(self):
        with pytest.raises(ValidationError, match="explicit DLQ routing"):
            ApplicationSettings(
                rabbitmq_token_queue_delivery_limit=5,
                token_queue_retry_schedule_seconds=(5, 10, 20, 40, 60),
            )

    def test_validate_rabbitmq_topology_names_reject_blank(self):
        with pytest.raises(ValidationError, match="must not be blank"):
            ApplicationSettings(rabbitmq_token_exchange_name="   ")


class TestProductionSecretValidation:
    """
    Production must never boot with a sample/placeholder credential — the
    combination of a public GitHub repo and a copy-pasted .env.example value
    is exactly the class of incident this validator exists to prevent.
    """

    def test_production_rejects_placeholder_jwt_secret(self):
        with pytest.raises(ValidationError, match="unique JWT secret"):
            ApplicationSettings(
                app_environment="production",
                jwt_secret_key="CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET",
                database_password="a-real-unique-password",
                rabbitmq_password="a-real-unique-password",
            )

    def test_production_rejects_short_jwt_secret(self):
        with pytest.raises(ValidationError, match="unique JWT secret"):
            ApplicationSettings(
                app_environment="production",
                jwt_secret_key="too-short",
                database_password="a-real-unique-password",
                rabbitmq_password="a-real-unique-password",
            )

    def test_production_rejects_sample_database_password(self):
        with pytest.raises(ValidationError, match="unique database_password"):
            ApplicationSettings(
                app_environment="production",
                jwt_secret_key="x" * 32,
                database_password="mypassword",
                rabbitmq_password="a-real-unique-password",
            )

    def test_production_rejects_sample_rabbitmq_password(self):
        with pytest.raises(ValidationError, match="unique rabbitmq_password"):
            ApplicationSettings(
                app_environment="production",
                jwt_secret_key="x" * 32,
                database_password="a-real-unique-password",
                rabbitmq_password="rmq_password",
            )

    def test_development_tolerates_sample_secrets(self):
        """The same sample values are fine outside production — that's the point."""
        settings = ApplicationSettings(
            app_environment="development",
            jwt_secret_key="CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET",
            database_password="mypassword",
            rabbitmq_password="rmq_password",
        )

        assert settings.app_environment == "development"


class TestApplicationSettingsProperties:
    """Computed URL properties unwrap SecretStr via get_secret_value()."""

    def test_database_url_property(self):
        settings = ApplicationSettings(
            database_user="testuser",
            database_password="testpass",
            database_host="testhost",
            database_port=5432,
            database_name="testdb",
        )

        assert (
            settings.database_url
            == "postgresql+asyncpg://testuser:testpass@testhost:5432/testdb"
        )

    def test_database_url_sync_property(self):
        settings = ApplicationSettings(
            database_user="testuser",
            database_password="testpass",
            database_host="testhost",
            database_port=5432,
            database_name="testdb",
        )

        assert (
            settings.database_url_sync
            == "postgresql://testuser:testpass@testhost:5432/testdb"
        )

    def test_effective_database_url_routes_through_pgbouncer_when_enabled(self):
        settings = ApplicationSettings(
            database_user="testuser",
            database_password="testpass",
            database_name="testdb",
            pgbouncer_enabled=True,
            pgbouncer_host="pgbouncer-host",
            pgbouncer_port=6432,
        )

        assert settings.effective_database_url == (
            "postgresql+asyncpg://testuser:testpass@pgbouncer-host:6432/testdb"
        )

    def test_effective_database_url_falls_back_to_direct_connection(self):
        settings = ApplicationSettings(
            database_user="testuser",
            database_password="testpass",
            database_host="direct-host",
            database_port=5432,
            database_name="testdb",
            pgbouncer_enabled=False,
        )

        assert settings.effective_database_url == settings.database_url
        assert "direct-host" in settings.effective_database_url

    def test_redis_url_without_password(self):
        settings = ApplicationSettings(
            redis_host="testhost", redis_port=6379, redis_db=0, redis_password=None
        )

        assert settings.redis_url == "redis://testhost:6379/0"

    def test_redis_url_with_password(self):
        settings = ApplicationSettings(
            redis_host="testhost",
            redis_port=6379,
            redis_db=0,
            redis_password="testpass",
        )

        assert settings.redis_url == "redis://:testpass@testhost:6379/0"

    def test_redis_token_counter_url_uses_isolated_db(self):
        settings = ApplicationSettings(
            redis_host="testhost",
            redis_port=6379,
            redis_token_counter_db=1,
            redis_password=None,
        )

        assert settings.redis_token_counter_url == "redis://testhost:6379/1"

    def test_broker_url_default_composition(self):
        settings = ApplicationSettings(
            celery_broker_url=None,
            rabbitmq_user="testuser",
            rabbitmq_password="testpass",
            rabbitmq_host="testhost",
            rabbitmq_port=5672,
            rabbitmq_vhost="/",
        )

        assert settings.broker_url == "amqp://testuser:testpass@testhost:5672/"

    def test_broker_url_prefers_explicit_override(self):
        custom_url = "amqp://override:pass@override-host:5672//"
        settings = ApplicationSettings(
            celery_broker_url=custom_url,
            rabbitmq_user="testuser",
            rabbitmq_password="testpass",
        )

        assert settings.broker_url == custom_url


class TestApplicationSettingsFromEnvironment:
    """Environment variables and .env override YAML defaults."""

    def test_load_from_environment_variables(self, monkeypatch):
        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FASTAPI_PORT", "9000")
        monkeypatch.setenv("DATABASE_HOST", "testdb")
        monkeypatch.setenv("DATABASE_USER", "testuser")
        monkeypatch.setenv("DATABASE_PASSWORD", "testpass")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "200")

        settings = ApplicationSettings()

        assert settings.app_name == "Test App"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.fastapi_port == 9000
        assert settings.database_host == "testdb"
        assert settings.database_user == "testuser"
        assert settings.database_password.get_secret_value() == "testpass"
        assert settings.rate_limit_requests_per_minute == 200

    def test_case_insensitive_env_vars(self, monkeypatch):
        monkeypatch.setenv("app_name", "Test App Case")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("log_level", "WARNING")
        monkeypatch.setenv("fastapi_port", "8080")

        settings = ApplicationSettings()

        assert settings.app_name == "Test App Case"
        assert settings.debug is False
        assert settings.log_level == "WARNING"
        assert settings.fastapi_port == 8080
