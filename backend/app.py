"""FastAPI 应用工厂。"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.route import api_router
from api.auth_route import auth_router
from api.admin_route import admin_router
from config.settings import get_settings
from database.db import init_database
from utils.exception import register_exception_handlers
from utils.logger import configure_logger


STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """在服务请求前初始化运行资源。

    参数:
        application: FastAPI 应用实例。

    生成:
        应用运行期间的空上下文。

    异常:
        RuntimeError: 数据库初始化失败时抛出。
    """
    configure_logger()
    init_database()
    yield


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。

    返回:
        已完成配置的 FastAPI 应用。

    异常:
        RuntimeError: 挂载子应用失败时抛出。
    """
    settings = get_settings()
    application = FastAPI(title=settings.app_name, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(application)
    application.include_router(api_router, prefix="/api")
    application.include_router(auth_router, prefix="/api")
    application.include_router(admin_router, prefix="/api")
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/favicon.ico")

    return application
