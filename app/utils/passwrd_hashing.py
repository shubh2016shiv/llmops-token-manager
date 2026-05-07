"""Password hashing utilities using bcrypt."""

import hashlib

import bcrypt


class PasswordHasher:
    """Simple password hashing utility."""

    @staticmethod
    def _normalize_password(password: str) -> bytes:
        """
        Normalize password input before bcrypt.

        bcrypt only accepts up to 72 bytes. We pre-hash using SHA-256 so the
        verifier works consistently for arbitrary-length passwords.
        """
        return hashlib.sha256(password.encode("utf-8")).digest()

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password as string

        """
        salt = bcrypt.gensalt()
        normalized = PasswordHasher._normalize_password(password)
        hashed = bcrypt.hashpw(normalized, salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plain text password
            hashed_password: Previously hashed password

        Returns:
            True if password matches, False otherwise

        """
        try:
            normalized = PasswordHasher._normalize_password(password)
            return bcrypt.checkpw(normalized, hashed_password.encode("utf-8"))
        except (ValueError, TypeError):
            # Return False for invalid hash formats instead of raising exception
            return False
