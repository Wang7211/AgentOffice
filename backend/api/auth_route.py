"""认证与用户接口路由。"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Header
from fastapi import Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import get_db
from database.tables import SysUser
from utils.auth import create_access_token
from utils.auth import hash_password
from utils.auth import verify_password
from utils.auth import verify_token
from utils.exception import ParameterException
from utils.exception import success_response


DEFAULT_AVATAR = "/static/default-avatar.jpg"

auth_router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: str | None = None


class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    nickname: str | None
    role: str
    avatar: str | None


def _get_token_from_header(authorization: str = Header(...)) -> str:
    """从 Authorization 头中提取 Bearer 令牌。"""
    if not authorization.startswith("Bearer "):
        raise ParameterException("认证头格式错误")
    return authorization[7:]


def get_current_user(
    request: Request,
    db_session: Session = Depends(get_db),
) -> SysUser:
    """从请求中解析当前用户（依赖注入）。"""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ParameterException("未提供认证令牌")
    token = auth_header[7:]
    payload = verify_token(token)
    if payload is None:
        raise ParameterException("令牌无效或已过期")
    user = db_session.query(SysUser).filter_by(
        id=payload.get("user_id"),
        is_delete=0,
        status=1,
    ).first()
    if user is None:
        raise ParameterException("用户不存在或已被禁用")
    return user


@auth_router.post("/login")
async def login(
    request: LoginRequest,
    db_session: Session = Depends(get_db),
) -> object:
    """用户登录。"""
    user = db_session.query(SysUser).filter_by(
        username=request.username,
        is_delete=0,
    ).first()
    if user is None:
        raise ParameterException("用户名或密码错误")
    if user.status != 1:
        raise ParameterException("账号已被禁用")
    if not verify_password(request.password, user.password):
        raise ParameterException("用户名或密码错误")
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )
    return success_response(TokenData(
        access_token=token,
        user_id=user.id,
        username=user.username,
        nickname=user.nickname,
        role=user.role,
        avatar=user.avatar or DEFAULT_AVATAR,
    ).model_dump())


@auth_router.post("/register")
async def register(
    request: RegisterRequest,
    db_session: Session = Depends(get_db),
) -> object:
    """用户注册。"""
    if len(request.username) < 3:
        raise ParameterException("用户名至少 3 个字符")
    if len(request.password) < 6:
        raise ParameterException("密码至少 6 个字符")
    exists = db_session.query(SysUser).filter_by(
        username=request.username,
        is_delete=0,
    ).first()
    if exists:
        raise ParameterException("用户名已存在")
    user = SysUser(
        username=request.username,
        password=hash_password(request.password),
        nickname=request.nickname or request.username,
        role="user",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )
    return success_response(TokenData(
        access_token=token,
        user_id=user.id,
        username=user.username,
        nickname=user.nickname,
        role=user.role,
        avatar=user.avatar or DEFAULT_AVATAR,
    ).model_dump())


@auth_router.get("/me")
async def get_me(
    current_user: SysUser = Depends(get_current_user),
) -> object:
    """获取当前用户信息。"""
    return success_response({
        "user_id": current_user.id,
        "username": current_user.username,
        "nickname": current_user.nickname,
        "role": current_user.role,
        "avatar": current_user.avatar or DEFAULT_AVATAR,
        "status": current_user.status,
        "create_time": current_user.create_time.isoformat(),
    })


@auth_router.put("/profile")
async def update_profile(
    nickname: str | None = None,
    current_user: SysUser = Depends(get_current_user),
    db_session: Session = Depends(get_db),
) -> object:
    """更新当前用户信息。"""
    if nickname:
        current_user.nickname = nickname
    db_session.commit()
    return success_response({
        "user_id": current_user.id,
        "username": current_user.username,
        "nickname": current_user.nickname,
        "role": current_user.role,
        "avatar": current_user.avatar or DEFAULT_AVATAR,
    })
