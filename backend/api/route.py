"""REST 与 SSE 接口路由。"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.chat import ChatCompletionRequest
from schemas.knowledge import KnowledgeSearchRequest
from services.chat_service import ChatService
from services.knowledge_service import KnowledgeService
from services.tool_service import get_tool_registry
from utils.exception import ParameterException
from utils.exception import success_response


api_router = APIRouter()


@api_router.get("/health")
async def health_check() -> dict[str, str]:
    """返回服务健康状态。

    返回:
        健康状态字典。

    异常:
        无。
    """
    return {"status": "ok"}


@api_router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    db_session: Session = Depends(get_db),
) -> StreamingResponse:
    """创建聊天补全。

    参数:
        request: 聊天补全请求。
        db_session: 数据库会话。

    返回:
        `stream` 为真时返回 SSE 流，否则返回单次 JSON 响应。

    异常:
        ParameterException: 请求校验失败时抛出。
    """
    chat_service = ChatService(db_session)
    if request.stream:
        event_iterator = chat_service.stream_chat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return StreamingResponse(event_iterator, media_type="text/event-stream")
    result = await chat_service.complete_chat(
        message=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
    )
    return success_response(result)


@api_router.get("/tool/list")
async def list_tools() -> object:
    """列出全部已启用工具。

    返回:
        包含工具规格的统一 JSON 响应。

    异常:
        无。
    """
    registry = get_tool_registry()
    tool_specs = [spec.__dict__ for spec in registry.list_specs()]
    return success_response(tool_specs)


@api_router.post("/knowledge/upload")
async def upload_knowledge(
    upload_file: UploadFile = File(...),
    db_session: Session = Depends(get_db),
) -> object:
    """上传并索引知识库文档。

    参数:
        upload_file: 上传的 PDF、TXT 或 DOCX 文件。
        db_session: 数据库会话。

    返回:
        包含文件元数据的统一 JSON 响应。

    异常:
        ParameterException: 文件不合法时抛出。
    """
    service = KnowledgeService(db_session)
    result = await service.upload_file(upload_file)
    return success_response(result)


@api_router.post("/knowledge/search")
async def search_knowledge(
    request: KnowledgeSearchRequest,
    db_session: Session = Depends(get_db),
) -> object:
    """检索本地知识库。

    参数:
        request: 检索请求。
        db_session: 数据库会话。

    返回:
        包含命中文档分片的统一 JSON 响应。

    异常:
        无。
    """
    service = KnowledgeService(db_session)
    return success_response(service.search(request.query, request.top_k))


@api_router.get("/chat/history")
async def chat_history(
    session_id: str | None = Query(default=None),
    db_session: Session = Depends(get_db),
) -> object:
    """返回聊天会话或消息记录。

    参数:
        session_id: 可选的聊天会话标识。
        db_session: 数据库会话。

    返回:
        包含历史数据的统一 JSON 响应。

    异常:
        无。
    """
    service = ChatService(db_session)
    return success_response(service.history(session_id=session_id))


# ─── Chat Session Management ──────────────────────────────


@api_router.get("/chat/sessions")
async def list_sessions(
    user_id: int = Query(default=1),
    keyword: str | None = Query(default=None),
    db_session: Session = Depends(get_db),
) -> object:
    """获取用户的会话列表。

    参数:
        user_id: 用户标识。
        keyword: 可选的搜索关键词。
        db_session: 数据库会话。
    """
    service = ChatService(db_session)
    return success_response(service.list_sessions(user_id=user_id, keyword=keyword))


class RenameSessionRequest(BaseModel):
    session_name: str


@api_router.put("/chat/sessions/{session_id}/rename")
async def rename_session(
    session_id: str,
    request: RenameSessionRequest,
    db_session: Session = Depends(get_db),
) -> object:
    """重命名会话。"""
    if not request.session_name.strip():
        raise ParameterException("会话名称不能为空")
    service = ChatService(db_session)
    return success_response(service.rename_session(session_id, request.session_name))


@api_router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db_session: Session = Depends(get_db),
) -> object:
    """软删除会话。"""
    service = ChatService(db_session)
    return success_response(service.delete_session(session_id))
