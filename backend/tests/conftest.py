"""共享的测试 fixtures 和配置。"""

import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

# ---------------------------------------------------------------------------
# 环境变量：测试期间使用本地规则模型，不依赖外部 API
# ---------------------------------------------------------------------------

DEFAULT_TEST_DATABASE_URL = (
    "mysql+pymysql://agentoffice:agentoffice123@127.0.0.1:3307/"
    "agentoffice_test?charset=utf8mb4"
)

_TEST_ENV_VARS = {
    "APP_ENV": "test",
    "MODEL_PROVIDER": "local",
    "DATABASE_URL": os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL),
    "REDIS_URL": "",
    "OPENAI_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    "QWEN_API_KEY": "",
    "JWT_SECRET_KEY": "test-secret-key-for-testing-purposes-only!",
}

for key, value in _TEST_ENV_VARS.items():
    os.environ[key] = value


@pytest.fixture(autouse=True)
def _setup_test_env(monkeypatch: MonkeyPatch) -> None:
    """自动为每个测试设置测试环境变量并清除 Settings 缓存。"""
    for key, value in _TEST_ENV_VARS.items():
        monkeypatch.setenv(key, value)
    # 清除缓存以使 Settings 重新从环境变量加载
    from config.settings import get_settings as _gs

    _gs.cache_clear()
    _gs()


@pytest.fixture(scope="session", autouse=True)
def _init_test_database() -> None:
    """Initialize the MySQL test database once for the test session."""
    from database.db import init_database

    init_database()


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """临时数据目录，避免测试污染真实文件系统。"""
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    (data_dir / "uploads").mkdir(parents=True)
    (data_dir / "vector_store").mkdir(parents=True)
    (project_root / "logs").mkdir(parents=True)

    # 让 Settings 将 PROJECT_ROOT 指向临时目录
    monkeypatch.setattr("config.settings.PROJECT_ROOT", project_root)
    # 清除 Settings 缓存使其重新从环境变量加载
    monkeypatch.setattr("config.settings.get_settings.cache_clear", lambda: None)
    from config.settings import get_settings as _get_settings

    _get_settings.cache_clear()
    settings = _get_settings()
    return data_dir


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """创建一个简短的 PDF 测试文件。"""
    import fitz

    file_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Hello AgentOffice PDF Test")
    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """创建一个 TXT 测试文件。"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello AgentOffice TXT Test\n第二行内容", encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# 模拟注册表工具
# ---------------------------------------------------------------------------


@pytest.fixture
def _clear_tool_registry_cache():
    """清除工具注册表缓存，避免测试间互相干扰。"""
    from services import tool_service

    tool_service.get_tool_registry.cache_clear()


@pytest.fixture
def _clear_settings_cache():
    """清除 Settings 缓存。"""
    from config import settings

    settings.get_settings.cache_clear()
