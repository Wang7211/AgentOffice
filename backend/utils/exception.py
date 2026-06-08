"""业务异常与 FastAPI 全局异常处理。"""

from enum import IntEnum

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import ORJSONResponse
from loguru import logger
from pydantic import ValidationError
from starlette.exceptions import HTTPException


class ErrorCode(IntEnum):
    """统一接口错误码。"""

    SUCCESS = 0
    PARAM_ERROR = 40001
    NOT_FOUND = 40401
    TOOL_ERROR = 50001
    MODEL_ERROR = 50002
    STORAGE_ERROR = 50003
    SYSTEM_ERROR = 50000


class BusinessException(Exception):
    """可预期业务失败的基础异常。

    参数:
        message: 面向用户的错误信息。
        code: 统一接口错误码。

    异常:
        无。
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.SYSTEM_ERROR,
    ) -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class ParameterException(BusinessException):
    """请求参数不合法时抛出的异常。"""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code=ErrorCode.PARAM_ERROR)


class ToolException(BusinessException):
    """工具执行失败时抛出的异常。"""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code=ErrorCode.TOOL_ERROR)


class ModelException(BusinessException):
    """模型生成失败时抛出的异常。"""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code=ErrorCode.MODEL_ERROR)


class StorageException(BusinessException):
    """数据持久化失败时抛出的异常。"""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code=ErrorCode.STORAGE_ERROR)


def success_response(data: object = None) -> ORJSONResponse:
    """构造统一成功响应。

    参数:
        data: 响应数据载荷。

    返回:
        符合项目统一格式的 JSON 响应。

    异常:
        TypeError: 数据无法序列化时抛出。
    """
    return ORJSONResponse(
        {"code": ErrorCode.SUCCESS, "message": "success", "data": data}
    )


def error_response(code: int, message: str, status_code: int = 200) -> ORJSONResponse:
    """构造统一错误响应。

    参数:
        code: 统一接口错误码。
        message: 面向用户的错误信息。

    返回:
        符合项目统一格式的 JSON 响应。

    异常:
        TypeError: 数据无法序列化时抛出。
    """
    return ORJSONResponse(
        {"code": code, "message": message, "data": None},
        status_code=status_code,
    )


def register_exception_handlers(application: FastAPI) -> None:
    """注册全局异常处理器。

    参数:
        application: FastAPI 应用实例。

    返回:
        无。

    异常:
        无。
    """

    @application.exception_handler(BusinessException)
    async def handle_business_exception(
        request: Request,
        exc: BusinessException,
    ) -> ORJSONResponse:
        logger.warning("business error: {}", exc.message)
        return error_response(code=exc.code, message=exc.message)

    @application.exception_handler(ValidationError)
    async def handle_validation_exception(
        request: Request,
        exc: ValidationError,
    ) -> ORJSONResponse:
        logger.warning("validation error: {}", exc.errors())
        return error_response(
            code=ErrorCode.PARAM_ERROR,
            message="请求参数格式错误",
        )

    @application.exception_handler(HTTPException)
    async def handle_http_exception(
        request: Request,
        exc: HTTPException,
    ) -> ORJSONResponse:
        logger.warning("http error: {}", exc.detail)
        return error_response(
            code=exc.status_code,
            message=str(exc.detail),
            status_code=exc.status_code,
        )

    @application.exception_handler(Exception)
    async def handle_system_exception(
        request: Request,
        exc: Exception,
    ) -> ORJSONResponse:
        logger.exception("system error: {}", exc)
        return error_response(
            code=ErrorCode.SYSTEM_ERROR,
            message="系统异常，请联系管理员",
        )
