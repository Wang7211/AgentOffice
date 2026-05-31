"""聊天请求与响应数据模型。"""

from pydantic import BaseModel
from pydantic import Field


class ChatCompletionRequest(BaseModel):
    """聊天补全请求模型。"""

    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    stream: bool = True
    user_id: int = 1


class ChatCompletionResponse(BaseModel):
    """聊天补全响应模型。"""

    session_id: str
    message_id: int
    answer: str
    tool_name: str | None = None
    tool_result: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    plan: list[dict[str, object]] | None = None
    reflection: dict[str, object] | None = None
    citations: list[dict[str, str]] | None = None


class ChatHistoryRequest(BaseModel):
    """聊天历史请求模型。"""

    session_id: str | None = None
