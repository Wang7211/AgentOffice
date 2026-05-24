"""MySQL 数据库引擎、会话和初始化辅助逻辑。"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings


class Base(DeclarativeBase):
    """ORM 实体基类。"""


def _build_engine_url() -> str:
    """构建 MySQL 数据库连接地址。

    返回:
        SQLAlchemy MySQL 数据库连接地址。
    异常:
        ValueError: 当前数据库地址不是 MySQL 地址时抛出。
    """
    settings = get_settings()
    database_url = settings.database_url
    if not _is_mysql_url(database_url):
        raise ValueError("当前项目已切换为 MySQL，请在 DATABASE_URL 中配置 MySQL 连接地址。")
    return database_url


def _is_mysql_url(database_url: str) -> bool:
    """判断当前连接地址是否为 MySQL。"""
    driver_name = make_url(database_url).drivername
    return driver_name.startswith("mysql")


def _ensure_mysql_database(database_url: str) -> None:
    """在 MySQL 数据库不存在时自动创建目标库。

    参数:
        database_url: 完整 MySQL 连接地址。
    返回:
        无。
    异常:
        sqlalchemy.exc.SQLAlchemyError: MySQL 服务不可用或账号无建库权限时抛出。
    """
    url = make_url(database_url)
    database_name = url.database
    if not database_name:
        return

    server_url = url.set(database=None)
    server_engine = create_engine(server_url, pool_pre_ping=True)
    quoted_database = database_name.replace("`", "``")
    with server_engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{quoted_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
    server_engine.dispose()


DATABASE_URL = _build_engine_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """为 FastAPI 依赖注入提供数据库会话。

    生成:
        SQLAlchemy 数据库会话。
    异常:
        sqlalchemy.exc.SQLAlchemyError: 数据库操作失败时抛出。
    """
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


def init_database() -> None:
    """在数据表不存在时创建数据库表。

    返回:
        无。
    异常:
        sqlalchemy.exc.SQLAlchemyError: 创建数据库表失败时抛出。
    """
    from database import tables

    _ensure_mysql_database(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    tables.seed_default_data(SessionLocal())
