"""JWT 认证工具。"""

from datetime import datetime
from datetime import timedelta

import bcrypt
import jwt

from config.settings import get_settings
from database.tables import SysUser


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 小时


def create_access_token(user_id: int, username: str, role: str) -> str:
    """创建 JWT 访问令牌。

    参数:
        user_id: 用户标识。
        username: 用户名。
        role: 用户角色。

    返回:
        JWT 编码后的令牌字符串。
    """
    settings = get_settings()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def verify_token(token: str) -> dict | None:
    """验证 JWT 令牌并返回载荷。

    参数:
        token: JWT 令牌字符串。

    返回:
        解码后的令牌载荷，验证失败返回 None。
    """
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码与哈希密码是否匹配。

    参数:
        plain_password: 明文密码。
        hashed_password: bcrypt 哈希密码。

    返回:
        密码匹配返回 True。
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def hash_password(password: str) -> str:
    """对明文密码进行 bcrypt 哈希。

    参数:
        password: 明文密码。

    返回:
        bcrypt 哈希后的密码字符串。
    """
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def get_user_from_token(
    token: str,
    db_session: object,
) -> SysUser | None:
    """从令牌中解析用户并查询数据库。

    参数:
        token: JWT 令牌。
        db_session: 数据库会话。

    返回:
        用户对象，令牌无效或用户不存在时返回 None。
    """
    payload = verify_token(token)
    if payload is None:
        return None
    from sqlalchemy.orm import Session
    from database.db import SessionLocal

    session = db_session if isinstance(db_session, Session) else SessionLocal()
    try:
        user = session.query(SysUser).filter_by(
            id=payload.get("user_id"),
            is_delete=0,
            status=1,
        ).first()
        return user
    finally:
        if session is not db_session:
            session.close()
