"""SQLAlchemy ORM 数据表定义。"""

from datetime import datetime

import bcrypt
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session
from sqlalchemy.orm import mapped_column

from database.db import Base
from utils.common import now_datetime


class SysUser(Base):
    """系统用户表，预留权限管理能力。"""

    __tablename__ = "sys_user"
    __table_args__ = (UniqueConstraint("username", name="uk_sys_user_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(32), nullable=False)
    password: Mapped[str] = mapped_column(String(128), nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(32), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    avatar: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_delete: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
        onupdate=now_datetime,
    )


class ChatSession(Base):
    """聊天会话表。"""

    __tablename__ = "chat_session"
    __table_args__ = (Index("idx_chat_session_user", "user_id", "is_delete"),)

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    session_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_delete: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
        onupdate=now_datetime,
    )


class ChatRecord(Base):
    """聊天消息记录表。"""

    __tablename__ = "chat_record"
    __table_args__ = (Index("idx_chat_record_session", "session_id", "is_delete"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_cost: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_delete: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )


class ToolRecord(Base):
    """工具执行日志表。"""

    __tablename__ = "tool_record"
    __table_args__ = (Index("idx_tool_record_chat", "chat_record_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_record_id: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_input: Mapped[str] = mapped_column(Text, nullable=False)
    tool_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )


class KnowledgeFile(Base):
    """知识库上传文件表。"""

    __tablename__ = "knowledge_file"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_suffix: Mapped[str] = mapped_column(String(16), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    save_path: Mapped[str] = mapped_column(String(255), nullable=False)
    upload_user_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_delete: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )


class KnowledgeChunk(Base):
    """知识库文本分片表。"""

    __tablename__ = "knowledge_chunk"
    __table_args__ = (
        Index("idx_knowledge_chunk_file", "file_id", "is_delete"),
        Index("idx_knowledge_chunk_order", "file_id", "chunk_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("knowledge_file.id"))
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    vector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    is_delete: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )


class SystemConfig(Base):
    """系统键值配置表。"""

    __tablename__ = "system_config"
    __table_args__ = (UniqueConstraint("config_key", name="uk_config_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_key: Mapped[str] = mapped_column(String(64), nullable=False)
    config_value: Mapped[str | None] = mapped_column(String(512), nullable=True)
    config_name: Mapped[str] = mapped_column(String(128), nullable=False)
    config_type: Mapped[str] = mapped_column(String(32), nullable=False)
    remark: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_delete: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=now_datetime,
        onupdate=now_datetime,
    )


def seed_default_data(db_session: Session) -> None:
    """写入初始管理员用户和默认配置。

    参数:
        db_session: SQLAlchemy 数据库会话。

    返回:
        无。

    异常:
        sqlalchemy.exc.SQLAlchemyError: 数据库写入失败时抛出。
    """
    try:
        has_admin = db_session.query(SysUser).filter_by(username="admin").first()
        if not has_admin:
            db_session.add(_build_admin_user())
        _seed_system_config(db_session)
        db_session.commit()
    finally:
        db_session.close()


def _build_admin_user() -> SysUser:
    """构造使用哈希密码的默认管理员用户。"""
    password_hash = bcrypt.hashpw(
        b"admin",
        bcrypt.gensalt(),
    ).decode("utf-8")
    return SysUser(
        username="admin",
        password=password_hash,
        nickname="Administrator",
        role="admin",
    )


def _seed_system_config(db_session: Session) -> None:
    """创建默认动态配置记录。"""
    default_configs = [
        ("model_name", "local-rule-model", "默认模型名称", "model"),
        ("chunk_size", "800", "知识库分片长度", "file"),
        ("similarity_threshold", "0.1", "相似度阈值", "vector"),
    ]
    for key, value, name, config_type in default_configs:
        exists_config = db_session.query(SystemConfig).filter_by(
            config_key=key,
        ).first()
        if exists_config:
            continue
        db_session.add(
            SystemConfig(
                config_key=key,
                config_value=value,
                config_name=name,
                config_type=config_type,
            )
        )
