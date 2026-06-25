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
from memory.store import redis_kv
from utils.common import generate_uuid
from utils.common import local_isoformat
from utils.exception import ParameterException
from utils.structured_log import log_agent_event
from utils.structured_log import preview_text


_SESSION_CACHE_TTL = 3600  # 会话缓存 1 小时


class _CachedSession:
    """Redis 会话缓存的轻量代理。"""

    def __init__(self, data: dict) -> None:
        self.session_id = data["session_id"]
        self.user_id = data.get("user_id")
        self.model_name = data.get("model_name", "")


def _cache_session(session: ChatSession) -> None:
    """将会话写入 Redis 缓存。"""
    redis_kv.set(
        f"session:{session.session_id}",
        {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "session_name": session.session_name,
            "model_name": session.model_name,
            "is_delete": session.is_delete,
            "update_time": (
                session.update_time.isoformat() if session.update_time else ""
            ),
        },
        ttl=_SESSION_CACHE_TTL,
    )


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
            recent_history = self._load_agent_history(session.session_id, user_id)
            recent_observations = self._load_recent_observations(
                session.session_id,
                user_id,
            )
            user_record = self._create_record(session.session_id, "user", message)
            log_agent_event(
                "chat_record_buffered",
                session_id=session.session_id,
                record_id=user_record.id,
                role="user",
                content_preview=preview_text(message),
            )
            chat_memory.append(session.session_id, "user", message)
            agent_result = self._agent_graph.run(
                user_message=message,
                session_id=session.session_id,
                history=recent_history,
                user_id=user_id,
                recent_observations=recent_observations,
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
            chat_memory.append_observations(
                session.session_id,
                self._tool_observations(agent_result),
            )
            chat_memory.append(session.session_id, "assistant", agent_result["answer"])
            self._db_session.commit()
            tool_name = self._primary_tool_name(agent_result)
            tool_observations = self._tool_observations(agent_result)
            observation_summary = self._observation_summary(agent_result)
            log_agent_event(
                "chat_committed",
                session_id=session.session_id,
                user_record_id=user_record.id,
                assistant_record_id=assistant_record.id,
                tool_name=tool_name or "direct_answer",
                tool_names=self._tool_names_from_observations(tool_observations),
                has_observation=bool(observation_summary),
                total_duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            citations = self._extract_citations(agent_result)
            return {
                "session_id": session.session_id,
                "message_id": assistant_record.id,
                "answer": agent_result["answer"],
                "tool_calls": agent_result.get("tool_calls", []),
                "observations": agent_result.get("observations", []),
                "plan": agent_result.get("plan", []),
                "reflection": agent_result.get("reflection", {}),
                "citations": citations,
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

    def _load_agent_history(
        self,
        session_id: str,
        user_id: int,
    ) -> list[dict[str, str]]:
        """Load the recent conversation context used by AgentGraph.

        ChatMemory is a process-local cache. When it is empty, rebuild the
        recent window from MySQL so existing sessions survive process restarts.
        """
        recent_history = chat_memory.get_recent(session_id)
        history_source = "memory"
        if not recent_history:
            recent_history = self._restore_recent_history(session_id, user_id)
            history_source = "database" if recent_history else "empty"

        summary = redis_kv.get(f"summary:{session_id}")

        log_agent_event(
            "chat_history_loaded",
            session_id=session_id,
            user_id=user_id,
            source=history_source,
            message_count=len(recent_history),
            has_summary=bool(summary),
        )
        return recent_history

    def _load_recent_observations(
        self,
        session_id: str,
        user_id: int,
    ) -> list[dict[str, object]]:
        """Load recent reusable tool results for follow-up references."""
        observations = chat_memory.get_recent_observations(session_id)
        source = "memory"
        if not observations:
            observations = self._restore_recent_observations(session_id, user_id)
            source = "database" if observations else "empty"
        log_agent_event(
            "recent_observations_loaded",
            session_id=session_id,
            user_id=user_id,
            source=source,
            observation_count=len(observations),
            tool_names=self._tool_names_from_observations(observations),
        )
        return observations

    def _restore_recent_observations(
        self,
        session_id: str,
        user_id: int,
    ) -> list[dict[str, object]]:
        """Restore recent tool observations from persisted ToolRecord rows."""
        records = (
            self._db_session.query(ToolRecord)
            .join(ChatRecord, ChatRecord.id == ToolRecord.chat_record_id)
            .join(ChatSession, ChatSession.session_id == ChatRecord.session_id)
            .filter(
                ChatRecord.session_id == session_id,
                ChatRecord.is_delete == 0,
                ChatSession.user_id == user_id,
                ChatSession.is_delete == 0,
            )
            .order_by(desc(ToolRecord.create_time), desc(ToolRecord.id))
            .limit(10)
            .all()
        )
        observations = [
            self._tool_record_to_observation(record)
            for record in reversed(records)
        ]
        chat_memory.append_observations(session_id, observations)
        return observations

    @staticmethod
    def _tool_record_to_observation(record: ToolRecord) -> dict[str, object]:
        try:
            tool_input = json.loads(record.tool_input or "{}")
        except json.JSONDecodeError:
            tool_input = {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        status = "success" if int(record.status or 0) == 1 else "failed"
        return {
            "type": "tool_result",
            "step_id": f"tool_record:{record.id}",
            "tool_record_id": record.id,
            "chat_record_id": record.chat_record_id,
            "tool_name": record.tool_name,
            "tool_input": tool_input,
            "content": record.tool_result or "",
            "status": status,
            "error_msg": record.error_msg or "",
            "duration_ms": round(float(record.cost_time or 0.0) * 1000, 2),
            "created_at": local_isoformat(record.create_time),
            "reused": True,
        }

    def _restore_recent_history(
        self,
        session_id: str,
        user_id: int,
    ) -> list[dict[str, str]]:
        """Restore the short-term chat window from persisted chat records."""
        settings = get_settings()
        max_messages = max(settings.chat_window_size * 2, 1)
        records = (
            self._db_session.query(ChatRecord)
            .join(ChatSession, ChatSession.session_id == ChatRecord.session_id)
            .filter(
                ChatRecord.session_id == session_id,
                ChatRecord.is_delete == 0,
                ChatRecord.role.in_(("user", "assistant")),
                ChatSession.user_id == user_id,
                ChatSession.is_delete == 0,
            )
            .order_by(desc(ChatRecord.create_time), desc(ChatRecord.id))
            .limit(max_messages)
            .all()
        )
        history = [
            {"role": record.role, "content": record.content}
            for record in reversed(records)
        ]
        for item in history:
            chat_memory.append(session_id, item["role"], item["content"])
        return history

    @staticmethod
    def _tool_observations(
        agent_result: dict[str, object],
    ) -> list[dict[str, object]]:
        observations = agent_result.get("observations") or []
        if not isinstance(observations, list):
            return []
        return [
            dict(item)
            for item in observations
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ]

    @staticmethod
    def _tool_names_from_observations(
        observations: list[dict[str, object]],
    ) -> list[str]:
        names: list[str] = []
        for observation in observations:
            tool_name = str(observation.get("tool_name") or "")
            if tool_name and tool_name not in names:
                names.append(tool_name)
        return names

    @staticmethod
    def _primary_tool_name(agent_result: dict[str, object]) -> str:
        for tool_call in ChatService._tool_observations(agent_result):
            tool_name = str(tool_call.get("tool_name") or "")
            if tool_name:
                return tool_name
        return ""

    @staticmethod
    def _observation_summary(agent_result: dict[str, object]) -> str:
        lines: list[str] = []
        for tool_call in ChatService._tool_observations(agent_result):
            if tool_call.get("status") != "success":
                continue
            tool_name = str(tool_call.get("tool_name") or "")
            content = str(tool_call.get("content") or "")
            if tool_name and content:
                lines.append(f"[{tool_name}]\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_citations(agent_result: dict[str, object]) -> list[dict[str, str]]:
        """从 agent 结果中提取知识库引用文档列表。"""
        tool_calls = (
            agent_result.get("observations") or []
        )
        if not isinstance(tool_calls, list) or not tool_calls:
            return []
        seen: set[str] = set()
        citations: list[dict[str, str]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            if tc.get("type") != "tool_result":
                continue
            if tc.get("tool_name") != "knowledge":
                continue
            metadata = tc.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            matches = metadata.get("matches") or []
            if not isinstance(matches, list):
                continue
            for match in matches:
                if not isinstance(match, dict):
                    continue
                file_name = str(match.get("metadata", {}).get("file_name", "") or "")
                if not file_name:
                    file_name = str(match.get("file_name", ""))
                if file_name and file_name not in seen:
                    seen.add(file_name)
                    citations.append({"file_name": file_name})
        return citations

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
        meta_data = json.dumps({
            "session_id": result["session_id"],
            "citations": result.get("citations", []),
            "tool_calls": result.get("tool_calls", []),
        })
        yield f"event: meta\ndata: {meta_data}\n\n"
        answer = str(result["answer"])
        # 逐字输出，实现打字机效果
        for char in answer:
            yield f"event: message\ndata: {json.dumps({'content': char})}\n\n"
        done_data = json.dumps({
            "citations": result.get("citations", []),
            "tool_calls": result.get("tool_calls", []),
        })
        yield f"event: done\ndata: {done_data}\n\n"

    def history(
        self,
        session_id: str | None = None,
        user_id: int = 1,
    ) -> list[dict[str, object]]:
        """返回聊天历史。

        参数:
            session_id: 可选的会话标识。

        返回:
            会话列表或消息列表。

        异常:
            无。
        """
        if session_id:
            session = self._db_session.query(ChatSession).filter_by(
                session_id=session_id,
                user_id=user_id,
                is_delete=0,
            ).first()
            if session is None:
                raise ParameterException("会话不存在")
            records = self._db_session.query(ChatRecord).filter_by(
                session_id=session_id,
                is_delete=0,
            ).order_by(ChatRecord.create_time.asc()).all()
            return [self._record_to_dict(record) for record in records]
        sessions = self._db_session.query(ChatSession).filter_by(
            user_id=user_id,
            is_delete=0,
        ).order_by(desc(ChatSession.update_time)).limit(50).all()
        return [self._session_to_dict(session) for session in sessions]

    def _get_or_create_session(
        self,
        session_id: str | None,
        user_id: int,
        message: str,
    ) -> ChatSession:
        """获取已有会话或创建新会话（带 Redis 缓存）。"""
        settings = get_settings()
        if session_id:
            # 1) Redis 缓存查询
            cached = redis_kv.get(f"session:{session_id}")
            if (
                cached
                and not cached.get("is_delete")
                and str(cached.get("user_id") or "") == str(user_id)
            ):
                return _CachedSession(cached)  # type: ignore[return-value]
            # 2) MySQL 兜底
            session = self._db_session.query(ChatSession).filter_by(
                session_id=session_id,
                user_id=user_id,
                is_delete=0,
            ).first()
            if session:
                _cache_session(session)
                return session
            raise ParameterException("会话不存在")
        new_session = ChatSession(
            session_id=generate_uuid(),
            user_id=user_id,
            session_name=message[:30],
            model_name=settings.model_name,
        )
        self._db_session.add(new_session)
        self._db_session.flush()
        _cache_session(new_session)
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
        tool_calls = (
            agent_result.get("observations")
        )
        if isinstance(tool_calls, list) and tool_calls:
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                if tool_call.get("type") != "tool_result":
                    continue
                self._create_single_tool_record(
                    chat_record_id=chat_record_id,
                    tool_name=str(tool_call.get("tool_name") or ""),
                    tool_input=tool_call.get("tool_input") or {},
                    tool_result=str(tool_call.get("content") or ""),
                    cost_time=float(tool_call.get("duration_ms") or 0) / 1000,
                    status=1 if tool_call.get("status") in ("success", "completed") else 0,
                    error_msg=str(tool_call.get("error_msg") or ""),
                )
            return
        log_agent_event(
            "tool_record_skipped",
            chat_record_id=chat_record_id,
            reason="no_real_tool_calls",
            agent_tool_name=self._primary_tool_name(agent_result),
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
            "create_time": local_isoformat(record.create_time),
        }

    def _session_to_dict(self, session: ChatSession) -> dict[str, object]:
        """将会话记录转换为字典。"""
        return {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "model_name": session.model_name,
            "create_time": local_isoformat(session.create_time),
            "update_time": local_isoformat(session.update_time),
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
        user_id: int = 1,
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
            user_id=user_id,
            is_delete=0,
        ).first()
        if session is None:
            raise ParameterException("会话不存在")
        session.session_name = session_name
        self._db_session.commit()
        redis_kv.delete(f"session:{session_id}")
        return self._session_to_dict(session)

    def delete_session(self, session_id: str, user_id: int = 1) -> dict[str, object]:
        """软删除会话。

        参数:
            session_id: 会话标识。

        返回:
            包含被删除会话标识的字典。
        """
        session = self._db_session.query(ChatSession).filter_by(
            session_id=session_id,
            user_id=user_id,
            is_delete=0,
        ).first()
        if session is None:
            raise ParameterException("会话不存在")
        session.is_delete = 1
        self._db_session.commit()
        redis_kv.delete(f"session:{session_id}")
        return {"session_id": session_id}
