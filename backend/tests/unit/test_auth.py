"""JWT 认证和密码哈希工具测试。"""

from datetime import datetime
from datetime import timezone

import bcrypt
import jwt
import pytest

from utils.auth import ALGORITHM
from utils.auth import create_access_token
from utils.auth import hash_password
from utils.auth import verify_password
from utils.auth import verify_token


class TestPasswordHashing:
    def test_hash_and_verify_match(self) -> None:
        hashed = hash_password("my_password123")
        assert verify_password("my_password123", hashed) is True

    def test_hash_and_verify_wrong(self) -> None:
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_hash_is_bcrypt(self) -> None:
        hashed = hash_password("test")
        assert hashed.startswith("$2b$")

    def test_different_hashes_for_same_password(self) -> None:
        """bcrypt 使用随机 salt，相同密码应产生不同的哈希。"""
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2


class TestJWTAuth:
    def test_create_and_verify_token(self) -> None:
        token = create_access_token(user_id=1, username="admin", role="admin")
        payload = verify_token(token)
        assert payload is not None
        assert payload["user_id"] == 1
        assert payload["username"] == "admin"
        assert payload["role"] == "admin"

    def test_token_contains_exp_and_iat(self) -> None:
        token = create_access_token(user_id=1, username="test", role="user")
        payload = verify_token(token)
        assert "exp" in payload
        assert "iat" in payload

    def test_verify_invalid_token_returns_none(self) -> None:
        payload = verify_token("invalid.jwt.token")
        assert payload is None

    def test_verify_expired_token_returns_none(self) -> None:
        from config.settings import get_settings

        settings = get_settings()
        expired_payload = {
            "user_id": 1,
            "username": "test",
            "role": "user",
            "exp": datetime.now(timezone.utc).timestamp() - 3600,
            "iat": datetime.now(timezone.utc).timestamp() - 7200,
        }
        expired_token = jwt.encode(
            expired_payload, settings.jwt_secret_key, algorithm=ALGORITHM
        )
        payload = verify_token(expired_token)
        assert payload is None

    def test_tampered_token_returns_none(self) -> None:
        token = create_access_token(user_id=1, username="test", role="user")
        tampered_token = token[:-5] + "XXXXX"
        payload = verify_token(tampered_token)
        assert payload is None
