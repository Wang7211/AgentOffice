"""聊天会话服务。"""

import json
import time
from collections.abc import AsyncIterator

from sqlalchemy import desc
from sqlalchemy.orm import Session

from agent.graph import AgentGraph
from config.settings import get_settings
from database.tables import ChatRecord
from database.tables import ChatSession
from database.tables import ToolRecord
from memory.store import chat_memory
from utils.common import generate_uuid
from utils.exception import ParameterException
from utils.structured_log import log_agent_event
from utils.structured_log import preview_text


class ChatService:
    """处理聊天会话与消息记录的服务。"""

    def __init__(self, db_session: Session) -> None:
        self._db_session = db_session
        self._agent_graph = AgentGraph()

    async def complete_chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: int = 1,
    ) -> dict[str, object]:
        """生成一次助手回复。

        参数:
            message: 用户消息。
            session_id: 已存在的会话标识。
            user_id: 用户标识。

        返回:
            助手回复与会话元数据。

        异常:
            ParameterException: 消息为空时抛出。
        """
        if not message.strip():
            raise ParameterException("消息内容不能为空")
        start_time = time.perf_counter()
        log_agent_event(
            "chat_request_received",
            input_session_id=session_id or "",
            user_id=user_id,
            message_preview=preview_text(message),
            message_length=len(message),
        )
        try:
            session = self._get_or_create_session(session_id, user_id, message)
            log_agent_event(
                "chat_session_ready",
                session_id=session.session_id,
                input_session_id=session_id or "",
                session_status=(
                    "reused"
                    if session_id and session.session_id == session_id
                    else "created"
                ),
                model_name=session.model_name,
            )
            user_record = self._create_record(session.session_id, "user", message)
            log_agent_event(
                "chat_record_buffered",
                session_id=session.session_id,
                record_id=user_record.id,
                role="user",
                content_preview=preview_text(message),
            )
            recent_history = chat_memory.get_recent(session.session_id)
            chat_memory.append(session.session_id, "user", message)
            agent_result = self._agent_graph.run(
                user_message=message,
                session_id=session.session_id,
                history=recent_history,
            )
            assistant_record = self._create_record(
                session.session_id,
                "assistant",
                agent_result["answer"],
            )
            log_agent_event(
                "chat_record_buffered",
                session_id=session.session_id,
                record_id=assistant_record.id,
                role="assistant",
                content_preview=preview_text(agent_result["answer"]),
            )
            self._create_tool_record(user_record.id, agent_result, start_time)
            chat_memory.append(session.session_id, "assistant", agent_result["answer"])
            self._db_session.commit()
            log_agent_event(
                "chat_committed",
                session_id=session.session_id,
                user_record_id=user_record.id,
                assistant_record_id=assistant_record.id,
                tool_name=agent_result.get("tool_name") or "direct_answer",
                has_tool_result=bool(agent_result.get("tool_result")),
                total_duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            return {
                "session_id": session.session_id,
                "message_id": assistant_record.id,
                "answer": agent_result["answer"],
                "tool_name": agent_result.get("tool_name"),
                "tool_result": agent_result.get("tool_result"),
                "tool_calls": agent_result.get("tool_calls", []),
                "plan": agent_result.get("plan", []),
                "reflection": agent_result.get("reflection", {}),
            }
        except Exception as exc:
            self._db_session.rollback()
            log_agent_event(
                "chat_failed",
                input_session_id=session_id or "",
                user_id=user_id,
                error_type=type(exc).__name__,
                error=str(exc),
                total_duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            raise

    async def stream_chat(
        self,
        message: str,
        session_id: str | None = None,
        user_id: int = 1,
    ) -> AsyncIterator[str]:
        """通过 SSE 流式返回一次助手回复。

        参数:
            message: 用户消息。
            session_id: 已存在的会话标识。
            user_id: 用户标识。

        生成:
            SSE 数据分片。

        异常:
            ParameterException: 消息为空时抛出。
        """
        result = await self.complete_chat(message, session_id, user_id)
        meta_data = json.dumps({"session_id": result["session_id"]})
        yield f"event: meta\ndata: {meta_data}\n\n"
        answer = str(result["answer"])
        # 逐字输出，实现打字机效果
        for char in answer:
            yield f"event: message\ndata: {json.dumps({'content': char})}\n\n"
        yield "event: done\ndata: {}\n\n"

    def history(self, session_id: str | None = None) -> list[dict[str, object]]:
        """返回聊天历史。

        参数:
            session_id: 可选的会话标识。

        返回:
            会话列表或消息列表。

        异常:
            无。
        """
        if session_id:
            records = self._db_session.query(ChatRecord).filter_by(
                session_id=session_id,
                is_delete=0,
            ).order_by(ChatRecord.create_time.asc()).all()
            return [self._record_to_dict(record) for record in records]
        sessions = self._db_session.query(ChatSession).filter_by(
            is_delete=0,
        ).order_by(desc(ChatSession.update_time)).limit(50).all()
        return [self._session_to_dict(session) for session in sessions]

    def _get_or_create_session(
        self,
        session_id: str | None,
        user_id: int,
        message: str,
    ) -> ChatSession:
        """获取已有会话或创建新会话。"""
        settings = get_settings()
        if session_id:
            session = self._db_session.get(ChatSession, session_id)
            if session and session.is_delete == 0:
                return session
        new_session = ChatSession(
            session_id=generate_uuid(),
            user_id=user_id,
            session_name=message[:30],
            model_name=settings.model_name,
        )
        self._db_session.add(new_session)
        self._db_session.flush()
        return new_session

    def _create_record(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> ChatRecord:
        """创建一条聊天记录。"""
        record = ChatRecord(
            session_id=session_id,
            role=role,
            content=content,
            token_cost=max(1, len(content) // 4),
        )
        self._db_session.add(record)
        self._db_session.flush()
        return record

    def _create_tool_record(
        self,
        chat_record_id: int,
        agent_result: dict[str, object],
        start_time: float,
    ) -> None:
        """在工具被调用时创建工具调用日志。"""
        tool_calls = agent_result.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                self._create_single_tool_record(
                    chat_record_id=chat_record_id,
                    tool_name=str(tool_call.get("tool_name") or ""),
                    tool_input=tool_call.get("tool_input") or {},
                    tool_result=str(
                        tool_call.get("tool_result") or tool_call.get("content") or "",
                    ),
                    cost_time=float(tool_call.get("duration_ms") or 0) / 1000,
                    status=1 if tool_call.get("status") == "success" else 0,
                    error_msg=str(tool_call.get("error_msg") or ""),
                )
            return
        log_agent_event(
            "tool_record_skipped",
            chat_record_id=chat_record_id,
            reason="no_real_tool_calls",
            agent_tool_name=str(agent_result.get("tool_name") or ""),
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )

    def _create_single_tool_record(
        self,
        chat_record_id: int,
        tool_name: str,
        tool_input: object,
        tool_result: str,
        cost_time: float,
        status: int,
        error_msg: str,
    ) -> None:
        """写入一条工具调用记录。"""
        if not tool_name:
            return
        tool_record = ToolRecord(
            chat_record_id=chat_record_id,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, ensure_ascii=False),
            tool_result=tool_result,
            cost_time=cost_time,
            status=status,
            error_msg=error_msg,
        )
        self._db_session.add(tool_record)
        log_agent_event(
            "tool_record_buffered",
            chat_record_id=chat_record_id,
            tool_name=tool_name,
            status=tool_record.status,
            cost_time=round(tool_record.cost_time, 4),
            result_preview=preview_text(tool_record.tool_result),
            error_preview=preview_text(tool_record.error_msg),
        )

    def _record_to_dict(self, record: ChatRecord) -> dict[str, object]:
        """将聊天记录转换为字典。"""
        return {
            "id": record.id,
            "session_id": record.session_id,
            "role": record.role,
            "content": record.content,
            "create_time": record.create_time.isoformat(),
        }

    def _session_to_dict(self, session: ChatSession) -> dict[str, object]:
        """将会话记录转换为字典。"""
        return {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "model_name": session.model_name,
            "create_time": session.create_time.isoformat(),
            "update_time": session.update_time.isoformat(),
        }

    def list_sessions(
        self,
        user_id: int = 1,
        keyword: str | None = None,
    ) -> list[dict[str, object]]:
        """获取用户的会话列表。

        参数:
            user_id: 用户标识。
            keyword: 可选的搜索关键词。

        返回:
            会话字典列表。
        """
        query = self._db_session.query(ChatSession).filter_by(
            user_id=user_id,
            is_delete=0,
        )
        if keyword:
            query = query.filter(ChatSession.session_name.like(f"%{keyword}%"))
        sessions = query.order_by(desc(ChatSession.update_time)).limit(100).all()
        return [self._session_to_dict(session) for session in sessions]

    def rename_session(
        self,
        session_id: str,
        session_name: str,
    ) -> dict[str, object]:
        """重命名会话。

        参数:
            session_id: 会话标识。
            session_name: 新名称。

        返回:
            更新后的会话字典。
        """
        session = self._db_session.query(ChatSession).filter_by(
            session_id=session_id,
            is_delete=0,
        ).first()
        if session is None:
            raise ParameterException("会话不存在")
        session.session_name = session_name
        self._db_session.commit()
        return self._session_to_dict(session)

    def delete_session(self, session_id: str) -> dict[str, object]:
        """软删除会话。

        参数:
            session_id: 会话标识。

        返回:
            包含被删除会话标识的字典。
        """
        session = self._db_session.query(ChatSession).filter_by(
            session_id=session_id,
            is_delete=0,
        ).first()
        if session is None:
            raise ParameterException("会话不存在")
        session.is_delete = 1
        self._db_session.commit()
        return {"session_id": session_id}
