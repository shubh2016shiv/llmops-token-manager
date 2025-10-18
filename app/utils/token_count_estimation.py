"""
================================================================================
LLM TOKEN COUNT ESTIMATION UTILITY - ARCHITECTURAL NOTES
================================================================================

PURPOSE:
--------
Token counting utility for LLM requests with DoS protection with fallback estimation.

CORE FEATURES:
--------------
• Multi-format input support (strings, chat messages, multimodal)
• Fallback estimation when primary method fails
• Input size validation (DoS prevention)
• Performance monitoring (processing time)

DEPENDENCIES:
-------------
• LiteLLM: Industry-standard token counting library for OpenAI, Anthropic, etc.
"""

import logging
import time
from typing import Union, List, Dict, Any
from ..models.token_manager_models import InputType, TokenEstimation

try:
    import litellm
except ImportError:
    raise ImportError("LiteLLM library required. Install with: pip install litellm")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TokenEstimator:
    """
    Production-grade token estimator for per-user requests.
    Uses LiteLLM for reliable, scalable estimation without external infrastructure.
    Includes input validation, fallbacks, and timeouts for resilience.
    """

    # Settings (configurable at class level for simplicity)
    _max_input_length = 1_000_000  # Reject inputs > 1M chars to prevent DoS
    _timeout_seconds = 5.0  # Max time for estimation

    @classmethod
    def configure(
        cls, max_input_length: int = 1_000_000, timeout_seconds: float = 5.0
    ) -> None:
        """
        Configure estimator settings.

        Args:
            max_input_length: Max input chars (DoS prevention)
            timeout_seconds: Max time for estimation
        """
        cls._max_input_length = max_input_length
        cls._timeout_seconds = timeout_seconds
        logger.info(
            f"TokenEstimator configured: max_input={max_input_length}, "
            f"timeout={timeout_seconds}s"
        )

    @staticmethod
    def _detect_input_type(input_data: Union[str, List[Dict[str, Any]]]) -> InputType:
        """Detect and validate input type."""
        if isinstance(input_data, str):
            return InputType.SIMPLE_STRING

        if not isinstance(input_data, list) or not input_data:
            return InputType.UNKNOWN

        # Validate message structure
        if not all(isinstance(msg, dict) and "role" in msg for msg in input_data):
            return InputType.UNKNOWN

        return InputType.CHAT_MESSAGES

    @staticmethod
    def _validate_input_size(
        input_data: Union[str, List[Dict[str, Any]]], max_length: int
    ) -> None:
        """
        Validate input size to prevent resource exhaustion.

        Raises:
            ValueError: If input exceeds size limit
        """
        if isinstance(input_data, str):
            length = len(input_data)
            if length > max_length:
                raise ValueError(
                    f"Input too large: {length} chars (max: {max_length}). "
                    "This prevents resource exhaustion."
                )
        else:
            # Estimate total content length in messages
            total_length = sum(len(str(msg.get("content", ""))) for msg in input_data)
            if total_length > max_length:
                raise ValueError(
                    f"Messages too large: ~{total_length} chars (max: {max_length})"
                )

    @staticmethod
    def _count_images(messages: List[Dict[str, Any]]) -> int:
        """Count images in messages (for logging/fallback)."""
        count = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                count += sum(
                    1
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") in ["image_url", "image"]
                )
        return count

    @classmethod
    def _fallback_estimate(
        cls,
        input_data: Union[str, List[Dict[str, Any]]],
        input_type: InputType,
        image_count: int,
    ) -> int:
        """
        Fallback estimation if LiteLLM fails.
        Uses conservative heuristics (~4 chars/token, 170 tokens/image).
        """
        logger.warning("Using fallback token estimation (LiteLLM unavailable)")

        # Estimate text tokens
        chars_per_token = 4
        if input_type == InputType.SIMPLE_STRING:
            text_tokens = len(input_data) // chars_per_token
        else:
            # Sum all text content
            total_chars = sum(len(str(msg.get("content", ""))) for msg in input_data)
            text_tokens = total_chars // chars_per_token

        # Add image tokens
        image_tokens = image_count * 170

        return text_tokens + image_tokens

    @classmethod
    def estimate(
        cls,
        input_data: Union[str, List[Dict[str, Any]]],
        model: str,
        safety_margin: float = 0.0,
    ) -> TokenEstimation:
        """
        Estimate tokens for a single per-user request with validation and resilience.

        Args:
            input_data: String prompt or list of message dictionaries
            model: Target LLM model
            safety_margin: Optional buffer (0.05 = 5%)

        Returns:
            TokenEstimation with detailed breakdown

        Raises:
            ValueError: If input is invalid or too large
        """
        start_time = time.time()

        # Step 1: Validate input type
        input_type = cls._detect_input_type(input_data)
        if input_type == InputType.UNKNOWN:
            raise ValueError(
                "Invalid input format. Expected string or list of message dicts "
                "with 'role' and 'content' fields."
            )

        # Step 2: Validate input size (DoS prevention)
        try:
            cls._validate_input_size(input_data, cls._max_input_length)
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        logger.info(f"Processing {input_type.value} for model: {model}")

        message_count = 0
        image_count = 0

        try:
            # Step 3: Estimate tokens using LiteLLM (with implicit timeout via overall check)
            if input_type == InputType.SIMPLE_STRING:
                total_tokens = litellm.token_counter(model=model, text=input_data)
                message_count = 1
                logger.info(f"✓ Simple string: {total_tokens} tokens")

            else:  # CHAT_MESSAGES
                total_tokens = litellm.token_counter(model=model, messages=input_data)
                message_count = len(input_data)
                image_count = cls._count_images(input_data)
                logger.info(
                    f"✓ Chat: {message_count} msgs, {image_count} images, "
                    f"{total_tokens} tokens"
                )

        except Exception as e:
            # Step 4: Fallback if LiteLLM fails
            logger.error(
                f"LiteLLM estimation failed: {e}. Using fallback.", exc_info=False
            )

            image_count = (
                cls._count_images(input_data)
                if input_type == InputType.CHAT_MESSAGES
                else 0
            )
            total_tokens = cls._fallback_estimate(input_data, input_type, image_count)
            message_count = (
                len(input_data) if input_type == InputType.CHAT_MESSAGES else 1
            )

            logger.warning(f"✓ Fallback estimate: {total_tokens} tokens (conservative)")

        # Step 5: Apply safety margin
        if safety_margin > 0:
            original = total_tokens
            total_tokens = int(total_tokens * (1 + safety_margin))
            logger.debug(
                f"Applied {safety_margin * 100:.1f}% margin ({original} → {total_tokens})"
            )

        # Step 6: Calculate metrics and timeout check
        processing_time = time.time() - start_time
        if processing_time > cls._timeout_seconds:
            logger.warning(
                f"Estimation took {processing_time:.2f}s (timeout: {cls._timeout_seconds}s)"
            )
        processing_time_ms = processing_time * 1000

        logger.info(f"✓ Complete: {total_tokens} tokens in {processing_time_ms:.2f}ms")

        return TokenEstimation(
            input_type=input_type,
            model=model,
            total_tokens=total_tokens,
            text_tokens=total_tokens,  # LiteLLM includes all
            image_tokens=0,  # Included in total
            tool_tokens=0,  # Included in total
            message_count=message_count,
            image_count=image_count,
            processing_time_ms=processing_time_ms,
        )


# Convenience function
def estimate_tokens(
    input_data: Union[str, List[Dict[str, Any]]], model: str, safety_margin: float = 0.0
) -> TokenEstimation:
    """Quick token estimation."""
    return TokenEstimator.estimate(input_data, model, safety_margin)
