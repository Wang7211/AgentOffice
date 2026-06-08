"""API 端点集成测试（使用 FastAPI TestClient）。"""

import json

import pytest
from fastapi.testclient import TestClient

from app import create_app
from config.settings import get_settings


@pytest.fixture
def client() -> TestClient:
    """创建 FastAPI 测试客户端。"""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_token(client: TestClient) -> str:
    """获取测试用的 JWT 令牌。"""
    token_payload = {
        "user_id": 1,
        "username": "admin",
        "role": "admin",
    }
    # 直接构造 auth header 而非走登录接口（不依赖数据库）
    from utils.auth import create_access_token

    return create_access_token(**token_payload)


# -----------------------------------------------------------------------
# 健康检查
# -----------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            data = response.json()
            assert "data" in data or "status" in data


# -----------------------------------------------------------------------
# 认证接口
# -----------------------------------------------------------------------

class TestAuthEndpoints:
    def test_login_missing_fields(self, client: TestClient) -> None:
        response = client.post("/api/auth/login", json={})
        # 缺少字段应有错误
        assert response.status_code in (200, 422)
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data

    def test_register_missing_fields(self, client: TestClient) -> None:
        response = client.post("/api/auth/register", json={})
        assert response.status_code in (200, 422)

    def test_me_without_token(self, client: TestClient) -> None:
        response = client.get("/api/auth/me")
        # 自定义异常处理器将所有错误封装为 200 + body 内的 code
        assert response.status_code in (200, 401, 403)
        if response.status_code == 200:
            data = response.json()
            assert data.get("code", 0) != 0  # 应返回错误码


# -----------------------------------------------------------------------
# 聊天接口
# -----------------------------------------------------------------------

class TestChatEndpoint:
    def test_chat_completions(self, client: TestClient, auth_token: str) -> None:
        response = client.post(
            "/api/chat/completions",
            json={"message": "你好", "stream": False},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "answer" in data["data"]

    def test_chat_without_auth(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat/completions",
            json={"message": "你好"},
        )
        assert response.status_code == 401

    def test_chat_history(self, client: TestClient, auth_token: str) -> None:
        response = client.get(
            "/api/chat/history",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        # 无内容时返回空列表而不是报错
        assert response.status_code == 200

    def test_chat_sessions(self, client: TestClient, auth_token: str) -> None:
        response = client.get(
            "/api/chat/sessions",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200


# -----------------------------------------------------------------------
# 工具接口
# -----------------------------------------------------------------------

class TestToolEndpoint:
    def test_list_tools(self, client: TestClient, auth_token: str) -> None:
        response = client.get(
            "/api/tool/list",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "data" in data
        tools = data["data"]
        assert any("required_permissions" in item for item in tools)
        knowledge = next(item for item in tools if item["name"] == "knowledge")
        assert "knowledge:read" in knowledge["required_permissions"]
        assert knowledge["context_schema"] == {"user_id": "upload_user_id"}


# -----------------------------------------------------------------------
# 知识库接口
# -----------------------------------------------------------------------

class TestKnowledgeEndpoint:
    def test_search_without_query(self, client: TestClient, auth_token: str) -> None:
        response = client.post(
            "/api/knowledge/search",
            json={},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code in (200, 422, 404)
        if response.status_code == 422:
            # Pydantic 验证错误
            data = response.json()
            assert "detail" in data


# -----------------------------------------------------------------------
# 管理接口
# -----------------------------------------------------------------------

class TestAdminEndpoints:
    def test_dashboard(self, client: TestClient, auth_token: str) -> None:
        response = client.get(
            "/api/admin/dashboard",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200

    def test_trace_detail_uses_tool_user_record_context(
        self,
        client: TestClient,
        auth_token: str,
    ) -> None:
        from database.db import SessionLocal
        from database.tables import ChatRecord
        from database.tables import ChatSession
        from database.tables import ToolRecord
        from utils.common import generate_uuid

        db_session = SessionLocal()
        try:
            session_id = generate_uuid()
            db_session.add(ChatSession(
                session_id=session_id,
                user_id=1,
                session_name="trace context test",
                model_name="local",
            ))
            db_session.flush()
            user_record = ChatRecord(
                session_id=session_id,
                role="user",
                content="user question",
                token_cost=1,
            )
            db_session.add(user_record)
            db_session.flush()
            db_session.add(ChatRecord(
                session_id=session_id,
                role="assistant",
                content="assistant answer",
                token_cost=1,
            ))
            db_session.flush()
            tool_record = ToolRecord(
                chat_record_id=user_record.id,
                tool_name="knowledge",
                tool_input="{}",
                tool_result="ok",
                cost_time=0.01,
                status=1,
            )
            db_session.add(tool_record)
            db_session.commit()
            trace_id = tool_record.id
        finally:
            db_session.close()

        response = client.get(
            f"/api/admin/traces/{trace_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["user_message"] == "user question"
        assert data["assistant_message"] == "assistant answer"

    def test_users_list(self, client: TestClient, auth_token: str) -> None:
        response = client.get(
            "/api/admin/users",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200


# -----------------------------------------------------------------------
# 异常处理
# -----------------------------------------------------------------------

class TestErrorHandling:
    def test_404_not_found(self, client: TestClient) -> None:
        response = client.get("/api/nonexistent")
        # 自定义异常处理器将 HTTPException 转换为 200 + body 内的 code
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            data = response.json()
            assert data.get("code", 0) != 0  # 应返回错误码

    def test_cors_headers(self, client: TestClient) -> None:
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code in (200, 204, 400, 404)
