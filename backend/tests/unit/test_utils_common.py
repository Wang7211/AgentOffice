"""辅助工具函数测试。"""

import pytest

from utils.common import generate_uuid
from utils.common import mask_sensitive
from utils.common import normalize_text
from utils.common import safe_file_name
from utils.common import split_text
from utils.common import text_hash


class TestNormalizeText:
    def test_removes_zero_width_space(self) -> None:
        assert normalize_text("ab​c") == "abc"

    def test_collapses_multiple_spaces(self) -> None:
        assert normalize_text("a   b    c") == "a b c"

    def test_collapses_excessive_newlines(self) -> None:
        assert normalize_text("a\n\n\nb") == "a\n\nb"

    def test_strips_whitespace(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""


class TestTextHash:
    def test_sha256_length(self) -> None:
        h = text_hash("hello")
        assert len(h) == 64  # SHA-256 hex digest

    def test_deterministic(self) -> None:
        assert text_hash("hello") == text_hash("hello")

    def test_different_inputs_differ(self) -> None:
        assert text_hash("hello") != text_hash("world")


class TestGenerateUUID:
    def test_length(self) -> None:
        uid = generate_uuid()
        assert len(uid) == 32

    def test_uniqueness(self) -> None:
        uuids = {generate_uuid() for _ in range(100)}
        assert len(uuids) == 100


class TestSafeFileName:
    def test_replaces_special_chars(self) -> None:
        assert safe_file_name("hello world?.txt") == "hello_world.txt"

    def test_preserves_normal_name(self) -> None:
        assert safe_file_name("report_2024.pdf") == "report_2024.pdf"

    def test_empty_stem_becomes_upload(self) -> None:
        name = safe_file_name(".txt")
        assert "txt" in name

    def test_mixed_unicode(self) -> None:
        name = safe_file_name("中文文件名称.pdf")
        assert name.endswith(".pdf")


class TestSplitText:
    def test_short_text_returns_single_chunk(self) -> None:
        chunks = split_text("hello world", chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_long_text_splits(self) -> None:
        text = "hello\n" * 200
        chunks = split_text(text, chunk_size=100)
        assert len(chunks) > 1

    def test_returns_at_least_one_chunk(self) -> None:
        chunks = split_text("")
        assert len(chunks) == 1 and chunks[0] == ""

    def test_chunk_respects_boundary(self) -> None:
        """每个分片应不超过 chunk_size（仅限单段落）。"""
        text = "a" * 1000
        chunks = split_text(text, chunk_size=300)
        assert all(len(c) <= 300 for c in chunks[:-1])


class TestMaskSensitive:
    def test_masks_password_key(self) -> None:
        result = mask_sensitive({"password": "secret123"})
        assert result["password"] == "***"

    def test_masks_token_key(self) -> None:
        result = mask_sensitive({"api_token": "abc123"})
        assert result["api_token"] == "***"

    def test_preserves_normal_keys(self) -> None:
        result = mask_sensitive({"username": "admin", "age": 30})
        assert result["username"] == "admin"
        assert result["age"] == 30

    def test_case_insensitive_masking(self) -> None:
        result = mask_sensitive({"API_KEY": "secret"})
        assert result["API_KEY"] == "***"
