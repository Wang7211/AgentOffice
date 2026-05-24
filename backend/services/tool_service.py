"""工具注册服务。"""

from functools import lru_cache

from loguru import logger

from services.mcp_service import discover_mcp_tools
from tools.base import ToolRegistry
from tools.browser_tool import BrowserTool
from tools.code_tool import CodeTool
from tools.file_tool import FileTool
from tools.knowledge_tool import KnowledgeTool
from tools.search_tool import SearchTool
from tools.time_tool import TimeTool


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    """构建并返回共享工具注册表。

    返回:
        包含全部内置工具的注册表。

    异常:
        ToolException: 工具重复注册时抛出。
    """
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(CodeTool())
    registry.register(FileTool())
    registry.register(TimeTool())
    registry.register(KnowledgeTool())
    registry.register(BrowserTool())
    for remote_tool in discover_mcp_tools():
        try:
            registry.register(remote_tool)
        except Exception as exc:
            logger.warning("MCP 远程工具注册失败：{} {}", remote_tool.name, exc)
    return registry
