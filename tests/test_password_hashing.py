"""
Unit tests for password hashing utilities
"""

from app.utils.passwrd_hashing import PasswordHasher


class TestPasswordHasher:
    """Test cases for PasswordHasher class"""

    def test_hash_password_basic(self):
        """Test basic password hashing functionality"""
        password = "test_password_123"
        hashed = PasswordHasher.hash_password(password)

        # Check that hash is not empty and is a string
        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) > 0

        # Check that hash is different from original password
        assert hashed != password

    def test_hash_password_empty_string(self):
        """Test hashing an empty password"""
        password = ""
        hashed = PasswordHasher.hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_password_special_characters(self):
        """Test hashing password with special characters"""
        password = "P@ssw0rd!#$%^&*()"
        hashed = PasswordHasher.hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_password_unicode(self):
        """Test hashing password with unicode characters"""
        password = "密码测试🔐"
        hashed = PasswordHasher.hash_password(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_password_consistency(self):
        """Test that hashing produces different results for same password (salt)"""
        password = "consistent_test_password"

        # Hash the same password multiple times
        hash1 = PasswordHasher.hash_password(password)
        hash2 = PasswordHasher.hash_password(password)

        # Should be different due to salt
        assert hash1 != hash2

        # But both should be valid for verification
        assert PasswordHasher.verify_password(password, hash1)
        assert PasswordHasher.verify_password(password, hash2)

    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "correct_password_123"
        hashed = PasswordHasher.hash_password(password)

        # Should verify correctly
        assert PasswordHasher.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "correct_password_123"
        wrong_password = "wrong_password_456"
        hashed = PasswordHasher.hash_password(password)

        # Should not verify with wrong password
        assert PasswordHasher.verify_password(wrong_password, hashed) is False

    def test_verify_password_empty_hashed(self):
        """Test verification with empty password"""
        password = ""
        hashed = PasswordHasher.hash_password(password)

        assert PasswordHasher.verify_password(password, hashed) is True
        assert PasswordHasher.verify_password("not_empty", hashed) is False

    def test_verify_password_special_characters(self):
        """Test verification with special characters"""
        password = "Sp€cial_P@ssw0rd!#$%"
        hashed = PasswordHasher.hash_password(password)

        assert PasswordHasher.verify_password(password, hashed) is True
        assert PasswordHasher.verify_password("Different_Password", hashed) is False

    def test_verify_password_unicode(self):
        """Test verification with unicode characters"""
        password = "测试密码🔐"
        hashed = PasswordHasher.hash_password(password)

        assert PasswordHasher.verify_password(password, hashed) is True
        assert PasswordHasher.verify_password("different_unicode", hashed) is False

    def test_hash_and_verify_roundtrip(self):
        """Test complete hash and verify roundtrip"""
        test_cases = [
            "simple_password",
            "Complex_P@ssw0rd!123",
            "",
            "密码",
            "🔐🔑",
            "a" * 1000,  # Long password
        ]

        for password in test_cases:
            hashed = PasswordHasher.hash_password(password)
            assert PasswordHasher.verify_password(password, hashed) is True

    def test_verify_with_different_hash_format(self):
        """Test that verification fails with completely different hash"""
        password = "test_password"
        fake_hash = "this_is_not_a_valid_bcrypt_hash"

        # Should return False for invalid hash format (graceful handling)
        assert PasswordHasher.verify_password(password, fake_hash) is False

    def test_hash_deterministic_per_call(self):
        """Test that hash_password is deterministic within a single call"""
        password = "deterministic_test"

        # Hash the password once and verify it
        hashed = PasswordHasher.hash_password(password)
        assert PasswordHasher.verify_password(password, hashed) is True

        # The hash should remain the same for verification purposes
        # (bcrypt hashes include salt, so same password = different hash, but each is valid)
        assert isinstance(hashed, str)
        assert hashed.startswith("$2b$") or hashed.startswith("$2y$")  # bcrypt format

    def test_password_case_sensitivity(self):
        """Test that password verification is case sensitive"""
        password = "CaseSensitive"
        hashed = PasswordHasher.hash_password(password)

        # Different cases should fail
        assert PasswordHasher.verify_password("casesensitive", hashed) is False
        assert PasswordHasher.verify_password("CaseSensitive", hashed) is True
        assert PasswordHasher.verify_password("CASESENSITIVE", hashed) is False
