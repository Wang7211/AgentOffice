"""工具数据模型。"""

from pydantic import BaseModel


class ToolInfo(BaseModel):
    """对外展示的工具信息。"""

    name: str
    description: str
    input_schema: dict[str, str]
