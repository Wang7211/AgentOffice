"""企业智能 Agent 服务启动入口。"""

import uvicorn

from app import create_app
from config.settings import get_settings


app = create_app()


def main() -> None:
    """启动 FastAPI 服务。

    返回:
        无。
    异常:
        RuntimeError: 当 uvicorn 无法启动服务时抛出。
    """
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    main()
