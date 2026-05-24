"""从环境变量加载应用配置。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# 项目根目录（backend 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """服务、存储、模型与工具的运行配置。

    属性:
        app_name: 接口文档展示的应用名称。
        app_env: 当前运行环境。
        app_host: uvicorn 监听地址。
        app_port: uvicorn 监听端口。
        app_reload: 是否启用 uvicorn 自动重载。
        database_url: SQLAlchemy 数据库连接地址。
        redis_url: Redis 连接地址。
        model_provider: 模型适配器使用的供应商名称。
        model_name: 会话中展示的默认模型名称。
        openai_api_key: 环境变量中的 OpenAI 密钥。
        openai_base_url: OpenAI 兼容接口地址。
        openai_model: OpenAI 默认模型名称。
        deepseek_api_key: 环境变量中的 DeepSeek 密钥。
        deepseek_base_url: DeepSeek 兼容接口地址。
        deepseek_model: DeepSeek 默认模型名称。
        qwen_api_key: 环境变量中的通义千问密钥。
        qwen_base_url: 通义千问 OpenAI 兼容接口地址。
        qwen_model: 通义千问默认模型名称。
        tavily_api_key: 环境变量中的 Tavily 搜索密钥。
        upload_dir: 知识库上传文件目录。
        vector_store_dir: 本地向量索引目录。
        max_upload_mb: 上传文件最大体积，单位 MB。
        chat_window_size: 短期记忆滑动窗口轮数。
        knowledge_similarity_threshold: 知识库 RAG 最低相似度。
        agent_memory_similarity_threshold: Agent 长期记忆最低相似度。
        request_timeout: 外部请求超时时间，单位秒。
        cors_origins: 允许跨域访问的来源列表。
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "AI Enterprise Agent"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False
    database_url: str = "mysql+pymysql://root:123456@127.0.0.1:3306/agentoffice?charset=utf8mb4"
    redis_url: str = "redis://localhost:6379/0"
    model_provider: str = "local"
    model_name: str = "local-rule-model"
    openai_api_key: str = Field(default="", repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    deepseek_api_key: str = Field(default="", repr=False)
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    qwen_api_key: str = Field(default="", repr=False)
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    tavily_api_key: str = Field(default="", repr=False)
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    max_upload_mb: int = 20
    chat_window_size: int = 10
    knowledge_similarity_threshold: float = 0.25
    agent_memory_similarity_threshold: float = 0.15
    request_timeout: float = 15.0
    cors_origins: list[str] = ["*"]
    mcp_http_endpoint: str = ""
    mcp_api_key: str = Field(default="", repr=False)
    jwt_secret_key: str = Field(
        default="agentoffice-secret-key-change-in-production",
        repr=False,
    )
    jwt_algorithm: str = "HS256"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存后的应用配置。

    返回:
        从环境变量和 `.env` 文件加载的配置对象。

    异常:
        pydantic.ValidationError: 环境变量值不合法时抛出。
    """
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    return settings
