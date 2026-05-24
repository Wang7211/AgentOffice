"""后台管理接口路由。"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy import desc
from sqlalchemy.orm import Session

from database.db import get_db
from database.tables import ChatRecord
from database.tables import ChatSession
from database.tables import KnowledgeChunk
from database.tables import KnowledgeFile
from database.tables import SysUser
from database.tables import SystemConfig
from database.tables import ToolRecord
from utils.auth import hash_password
from utils.exception import ParameterException
from utils.exception import success_response
from utils.structured_log import log_agent_event


admin_router = APIRouter(prefix="/admin", tags=["admin"])


# ─── Dashboard ─────────────────────────────────────────────


@admin_router.get("/dashboard")
async def dashboard_stats(
    db_session: Session = Depends(get_db),
) -> object:
    """获取后台 Dashboard 统计概览。"""
    user_count = db_session.query(func.count(SysUser.id)).filter_by(
        is_delete=0,
    ).scalar() or 0
    session_count = db_session.query(func.count(ChatSession.session_id)).filter_by(
        is_delete=0,
    ).scalar() or 0
    file_count = db_session.query(func.count(KnowledgeFile.id)).filter_by(
        is_delete=0,
    ).scalar() or 0
    chunk_count = db_session.query(func.count(KnowledgeChunk.id)).filter_by(
        is_delete=0,
    ).scalar() or 0
    tool_call_count = db_session.query(func.count(ToolRecord.id)).scalar() or 0
    today_chat_count = db_session.query(func.count(ChatRecord.id)).filter(
        func.date(ChatRecord.create_time) == func.current_date(),
    ).scalar() or 0
    recent_sessions = db_session.query(ChatSession).filter_by(
        is_delete=0,
    ).order_by(desc(ChatSession.update_time)).limit(10).all()
    recent_tools = db_session.query(ToolRecord).order_by(
        desc(ToolRecord.create_time),
    ).limit(10).all()
    return success_response({
        "user_count": user_count,
        "session_count": session_count,
        "file_count": file_count,
        "chunk_count": chunk_count,
        "tool_call_count": tool_call_count,
        "today_chat_count": today_chat_count,
        "recent_sessions": [
            {
                "session_id": s.session_id,
                "session_name": s.session_name,
                "create_time": s.create_time.isoformat(),
            }
            for s in recent_sessions
        ],
        "recent_tools": [
            {
                "id": t.id,
                "tool_name": t.tool_name,
                "status": t.status,
                "cost_time": t.cost_time,
                "create_time": t.create_time.isoformat(),
            }
            for t in recent_tools
        ],
    })


# ─── User Management ──────────────────────────────────────


@admin_router.get("/users")
async def list_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    keyword: str | None = Query(default=None),
    db_session: Session = Depends(get_db),
) -> object:
    """分页查询用户列表。"""
    query = db_session.query(SysUser).filter_by(is_delete=0)
    if keyword:
        query = query.filter(SysUser.username.like(f"%{keyword}%"))
    total = query.count()
    users = query.order_by(desc(SysUser.create_time)).offset(
        (page - 1) * page_size,
    ).limit(page_size).all()
    return success_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": u.id,
                "username": u.username,
                "nickname": u.nickname,
                "role": u.role,
                "avatar": u.avatar or "/static/default-avatar.jpg",
                "status": u.status,
                "create_time": u.create_time.isoformat(),
            }
            for u in users
        ],
    })


class UpdateUserRequest(BaseModel):
    nickname: str | None = None
    role: str | None = None
    status: int | None = None
    password: str | None = None


@admin_router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    request: UpdateUserRequest,
    db_session: Session = Depends(get_db),
) -> object:
    """更新用户信息。"""
    user = db_session.query(SysUser).filter_by(
        id=user_id, is_delete=0,
    ).first()
    if user is None:
        raise ParameterException("用户不存在")
    if request.nickname is not None:
        user.nickname = request.nickname
    if request.role is not None:
        user.role = request.role
    if request.status is not None:
        user.status = request.status
    if request.password:
        user.password = hash_password(request.password)
    db_session.commit()
    return success_response({"id": user.id})


@admin_router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db_session: Session = Depends(get_db),
) -> object:
    """软删除用户。"""
    user = db_session.query(SysUser).filter_by(
        id=user_id, is_delete=0,
    ).first()
    if user is None:
        raise ParameterException("用户不存在")
    user.is_delete = 1
    db_session.commit()
    return success_response({"id": user_id})


# ─── Knowledge File Management ────────────────────────────


@admin_router.get("/knowledge/files")
async def list_knowledge_files(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db_session: Session = Depends(get_db),
) -> object:
    """分页查询知识库文件列表（含分片数量）。"""
    query = db_session.query(
        KnowledgeFile,
        func.count(KnowledgeChunk.id).label("chunk_count"),
    ).outerjoin(
        KnowledgeChunk,
        KnowledgeFile.id == KnowledgeChunk.file_id,
    ).filter(
        KnowledgeFile.is_delete == 0,
    ).group_by(KnowledgeFile.id).order_by(
        desc(KnowledgeFile.create_time),
    )
    total = db_session.query(func.count(KnowledgeFile.id)).filter_by(
        is_delete=0,
    ).scalar() or 0
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return success_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": f.KnowledgeFile.id,
                "file_name": f.KnowledgeFile.file_name,
                "file_suffix": f.KnowledgeFile.file_suffix,
                "file_size": f.KnowledgeFile.file_size,
                "chunk_count": f.chunk_count,
                "create_time": f.KnowledgeFile.create_time.isoformat(),
            }
            for f in items
        ],
    })


@admin_router.get("/knowledge/files/{file_id}/chunks")
async def list_file_chunks(
    file_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db_session: Session = Depends(get_db),
) -> object:
    """分页查询文件的文本分片。"""
    file_record = db_session.query(KnowledgeFile).filter_by(
        id=file_id, is_delete=0,
    ).first()
    if file_record is None:
        raise ParameterException("文件不存在")
    query = db_session.query(KnowledgeChunk).filter_by(
        file_id=file_id,
        is_delete=0,
    ).order_by(KnowledgeChunk.chunk_index)
    total = query.count()
    chunks = query.offset((page - 1) * page_size).limit(page_size).all()
    return success_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "file_name": file_record.file_name,
        "items": [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text,
                "vector_id": c.vector_id,
                "create_time": c.create_time.isoformat(),
            }
            for c in chunks
        ],
    })


@admin_router.delete("/knowledge/files/{file_id}")
async def delete_knowledge_file(
    file_id: int,
    db_session: Session = Depends(get_db),
) -> object:
    """软删除知识库文件及其分片。"""
    file_record = db_session.query(KnowledgeFile).filter_by(
        id=file_id, is_delete=0,
    ).first()
    if file_record is None:
        raise ParameterException("文件不存在")
    file_record.is_delete = 1
    db_session.query(KnowledgeChunk).filter_by(file_id=file_id).update(
        {"is_delete": 1},
    )
    db_session.commit()
    return success_response({"id": file_id})


# ─── Trace / Tool Records ─────────────────────────────────


@admin_router.get("/traces")
async def list_traces(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    tool_name: str | None = Query(default=None),
    status: int | None = Query(default=None),
    db_session: Session = Depends(get_db),
) -> object:
    """分页查询工具调用记录（链路追踪）。"""
    query = db_session.query(ToolRecord).order_by(
        desc(ToolRecord.create_time),
    )
    if tool_name:
        query = query.filter(ToolRecord.tool_name == tool_name)
    if status is not None:
        query = query.filter(ToolRecord.status == status)
    total = query.count()
    records = query.offset((page - 1) * page_size).limit(page_size).all()
    trace_data = []
    for r in records:
        chat_record = db_session.query(ChatRecord).filter_by(id=r.chat_record_id).first()
        session = None
        if chat_record:
            session = db_session.query(ChatSession).filter_by(
                session_id=chat_record.session_id,
            ).first()
        trace_data.append({
            "id": r.id,
            "chat_record_id": r.chat_record_id,
            "tool_name": r.tool_name,
            "tool_input": r.tool_input,
            "tool_result": r.tool_result,
            "cost_time": r.cost_time,
            "status": r.status,
            "error_msg": r.error_msg,
            "create_time": r.create_time.isoformat(),
            "session_name": session.session_name if session else None,
        })
    return success_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": trace_data,
    })


@admin_router.get("/traces/{trace_id}")
async def get_trace_detail(
    trace_id: int,
    db_session: Session = Depends(get_db),
) -> object:
    """获取单条工具调用详情。"""
    record = db_session.query(ToolRecord).filter_by(id=trace_id).first()
    if record is None:
        raise ParameterException("记录不存在")
    chat_record = db_session.query(ChatRecord).filter_by(
        id=record.chat_record_id,
    ).first()
    session = None
    if chat_record:
        session = db_session.query(ChatSession).filter_by(
            session_id=chat_record.session_id,
        ).first()
    user_message = None
    assistant_message = None
    if chat_record:
        user_msg = db_session.query(ChatRecord).filter(
            ChatRecord.session_id == chat_record.session_id,
            ChatRecord.id < chat_record.id,
            ChatRecord.role == "user",
        ).order_by(desc(ChatRecord.id)).first()
        if user_msg:
            user_message = user_msg.content
        assistant_message = chat_record.content
    return success_response({
        "id": record.id,
        "tool_name": record.tool_name,
        "tool_input": record.tool_input,
        "tool_result": record.tool_result,
        "cost_time": record.cost_time,
        "status": record.status,
        "error_msg": record.error_msg,
        "create_time": record.create_time.isoformat(),
        "session_name": session.session_name if session else None,
        "session_id": session.session_id if session else None,
        "user_message": user_message,
        "assistant_message": assistant_message,
    })


# ─── System Config ─────────────────────────────────────────


@admin_router.get("/config")
async def list_config(
    db_session: Session = Depends(get_db),
) -> object:
    """列出所有系统配置。"""
    configs = db_session.query(SystemConfig).filter_by(is_delete=0).order_by(
        SystemConfig.config_type,
        SystemConfig.id,
    ).all()
    return success_response([
        {
            "id": c.id,
            "config_key": c.config_key,
            "config_value": c.config_value,
            "config_name": c.config_name,
            "config_type": c.config_type,
            "remark": c.remark,
        }
        for c in configs
    ])


class UpdateConfigRequest(BaseModel):
    id: int
    config_value: str


@admin_router.put("/config")
async def update_config(
    request: UpdateConfigRequest,
    db_session: Session = Depends(get_db),
) -> object:
    """更新单条系统配置。"""
    config = db_session.query(SystemConfig).filter_by(
        id=request.id, is_delete=0,
    ).first()
    if config is None:
        raise ParameterException("配置项不存在")
    config.config_value = request.config_value
    db_session.commit()
    log_agent_event(
        "config_updated",
        config_id=config.id,
        config_key=config.config_key,
    )
    return success_response({"id": config.id})
