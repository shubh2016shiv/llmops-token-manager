"""
Logger Setup
-----------
Centralized logging configuration using loguru.
Provides structured logging with proper formatting and rotation.
"""

import sys
from loguru import logger
from app.core.config_manager import settings


def configure_logger() -> None:
    """
    Configure loguru logger with appropriate settings.
    Removes default handler and adds custom formatted handler.
    """
    # Remove default handler
    logger.remove()

    # Add custom handler with formatting
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    # Add file handler for production
    if not settings.debug:
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            rotation="500 MB",
            retention="10 days",
            level=settings.log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            backtrace=True,
            diagnose=False,
        )

    logger.info(f"Logger configured with level: {settings.log_level}")


# Configure logger on import
configure_logger()
