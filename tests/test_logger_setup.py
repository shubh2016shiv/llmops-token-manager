"""
Comprehensive Unit Tests for Logger Setup
========================================
Unit tests for the centralized logging configuration using loguru.

Test Coverage:
- Basic configuration (3 tests)
- File handler configuration (2 tests)
- Log levels (4 tests)
- Format verification (2 tests)
- Integration (1 test)

Total: 12 comprehensive unit tests
"""

import sys
from unittest.mock import patch, MagicMock

from app.core.logger_setup import configure_logger


class TestLoggerSetup:
    """Test cases for logger configuration."""

    # Group 1: Basic Configuration (3 tests)

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_removes_default_handler(self, mock_settings, mock_logger):
        """Test that configure_logger removes default handler."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        mock_logger.remove.assert_called_once()

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_adds_stdout_handler_debug_mode(
        self, mock_settings, mock_logger
    ):
        """Test stdout handler configuration in debug mode."""
        # Setup
        mock_settings.log_level = "DEBUG"
        mock_settings.debug = True
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        mock_logger.add.assert_called_once()
        call_args = mock_logger.add.call_args

        # Verify stdout handler parameters
        assert call_args[0][0] == sys.stdout
        assert call_args[1]["level"] == "DEBUG"
        assert call_args[1]["colorize"] is True
        assert call_args[1]["backtrace"] is True
        assert call_args[1]["diagnose"] is True

        # Verify format contains required fields
        format_string = call_args[1]["format"]
        assert "{time:YYYY-MM-DD HH:mm:ss.SSS}" in format_string
        assert "{level: <8}" in format_string
        assert "{name}" in format_string
        assert "{function}" in format_string
        assert "{line}" in format_string
        assert "{message}" in format_string

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_adds_stdout_handler_production_mode(
        self, mock_settings, mock_logger
    ):
        """Test stdout handler configuration in production mode."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        # Should be called twice: stdout + file handler
        assert mock_logger.add.call_count == 2

        # Check first call (stdout handler)
        stdout_call = mock_logger.add.call_args_list[0]
        assert stdout_call[0][0] == sys.stdout
        assert stdout_call[1]["level"] == "INFO"
        assert stdout_call[1]["colorize"] is True
        assert stdout_call[1]["backtrace"] is True
        assert stdout_call[1]["diagnose"] is False

    # Group 2: File Handler Configuration (2 tests)

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_adds_file_handler_production(
        self, mock_settings, mock_logger
    ):
        """Test file handler configuration in production mode."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        # Should be called twice: stdout + file handler
        assert mock_logger.add.call_count == 2

        # Check second call (file handler)
        file_call = mock_logger.add.call_args_list[1]
        assert file_call[0][0] == "logs/app_{time:YYYY-MM-DD}.log"
        assert file_call[1]["rotation"] == "500 MB"
        assert file_call[1]["retention"] == "10 days"
        assert file_call[1]["level"] == "INFO"
        assert file_call[1]["backtrace"] is True
        assert file_call[1]["diagnose"] is False

        # Verify file format doesn't have color codes
        file_format = file_call[1]["format"]
        assert "<green>" not in file_format
        assert "<level>" not in file_format
        assert "<cyan>" not in file_format

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_no_file_handler_debug_mode(
        self, mock_settings, mock_logger
    ):
        """Test that no file handler is added in debug mode."""
        # Setup
        mock_settings.log_level = "DEBUG"
        mock_settings.debug = True
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        # Should be called only once (stdout handler only)
        assert mock_logger.add.call_count == 1

    # Group 3: Log Levels (4 tests)

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_with_trace_level(self, mock_settings, mock_logger):
        """Test logger configuration with TRACE level."""
        # Setup
        mock_settings.log_level = "TRACE"
        mock_settings.debug = True
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        call_args = mock_logger.add.call_args
        assert call_args[1]["level"] == "TRACE"

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_with_info_level(self, mock_settings, mock_logger):
        """Test logger configuration with INFO level."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        call_args = mock_logger.add.call_args_list[0]  # stdout handler
        assert call_args[1]["level"] == "INFO"

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_with_warning_level(self, mock_settings, mock_logger):
        """Test logger configuration with WARNING level."""
        # Setup
        mock_settings.log_level = "WARNING"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        call_args = mock_logger.add.call_args_list[0]  # stdout handler
        assert call_args[1]["level"] == "WARNING"

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_with_error_level(self, mock_settings, mock_logger):
        """Test logger configuration with ERROR level."""
        # Setup
        mock_settings.log_level = "ERROR"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        call_args = mock_logger.add.call_args_list[0]  # stdout handler
        assert call_args[1]["level"] == "ERROR"

    # Group 4: Format Verification (2 tests)

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_stdout_format_contains_required_fields(
        self, mock_settings, mock_logger
    ):
        """Test that stdout format contains all required fields."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = True
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        call_args = mock_logger.add.call_args
        format_string = call_args[1]["format"]

        # Check for required format fields
        assert "{time:YYYY-MM-DD HH:mm:ss.SSS}" in format_string
        assert "{level: <8}" in format_string
        assert "{name}" in format_string
        assert "{function}" in format_string
        assert "{line}" in format_string
        assert "{message}" in format_string

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_file_format_no_color_codes(
        self, mock_settings, mock_logger
    ):
        """Test that file format doesn't contain color codes."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        # Check file handler format (second call)
        file_call = mock_logger.add.call_args_list[1]
        file_format = file_call[1]["format"]

        # Should not contain color codes
        assert "<green>" not in file_format
        assert "<level>" not in file_format
        assert "<cyan>" not in file_format

        # Should still contain required fields
        assert "{time:YYYY-MM-DD HH:mm:ss.SSS}" in file_format
        assert "{level: <8}" in file_format
        assert "{name}" in file_format
        assert "{function}" in file_format
        assert "{line}" in file_format
        assert "{message}" in file_format

    # Group 5: Integration (1 test)

    @patch("app.core.logger_setup.logger")
    @patch("app.core.logger_setup.settings")
    def test_configure_logger_logs_configuration_message(
        self, mock_settings, mock_logger
    ):
        """Test that configuration message is logged."""
        # Setup
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        mock_logger.remove = MagicMock()
        mock_logger.add = MagicMock()
        mock_logger.info = MagicMock()

        # Act
        configure_logger()

        # Assert
        mock_logger.info.assert_called_once_with("Logger configured with level: INFO")
