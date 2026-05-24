"""异常体系和错误码测试。"""

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

import utils.exception
from utils.exception import BusinessException
from utils.exception import ErrorCode
from utils.exception import ModelException
from utils.exception import ParameterException
from utils.exception import StorageException
from utils.exception import ToolException
from utils.exception import error_response
from utils.exception import register_exception_handlers
from utils.exception import success_response


class TestErrorCode:
    def test_values(self) -> None:
        assert ErrorCode.SUCCESS == 0
        assert ErrorCode.PARAM_ERROR == 40001
        assert ErrorCode.NOT_FOUND == 40401
        assert ErrorCode.TOOL_ERROR == 50001
        assert ErrorCode.MODEL_ERROR == 50002
        assert ErrorCode.STORAGE_ERROR == 50003
        assert ErrorCode.SYSTEM_ERROR == 50000


class TestExceptionHierarchy:
    def test_business_exception_default_code(self) -> None:
        exc = BusinessException("出错了")
        assert exc.message == "出错了"
        assert exc.code == ErrorCode.SYSTEM_ERROR

    def test_parameter_exception(self) -> None:
        exc = ParameterException("参数错误")
        assert exc.code == ErrorCode.PARAM_ERROR

    def test_tool_exception(self) -> None:
        exc = ToolException("工具失败")
        assert exc.code == ErrorCode.TOOL_ERROR

    def test_model_exception(self) -> None:
        exc = ModelException("模型超时")
        assert exc.code == ErrorCode.MODEL_ERROR

    def test_storage_exception(self) -> None:
        exc = StorageException("写入失败")
        assert exc.code == ErrorCode.STORAGE_ERROR


class TestResponseHelpers:
    def test_success_response(self) -> None:
        response = success_response(data={"key": "value"})
        assert isinstance(response, ORJSONResponse)
        import json

        body = json.loads(response.body)
        assert body["code"] == 0
        assert body["message"] == "success"
        assert body["data"] == {"key": "value"}

    def test_success_response_none_data(self) -> None:
        response = success_response()
        import json

        body = json.loads(response.body)
        assert body["data"] is None

    def test_error_response(self) -> None:
        response = error_response(code=50001, message="工具错误")
        import json

        body = json.loads(response.body)
        assert body["code"] == 50001
        assert body["message"] == "工具错误"
        assert body["data"] is None


class TestRegisterExceptionHandlers:
    def test_registers_handlers(self) -> None:
        app = FastAPI()
        register_exception_handlers(app)
        # 验证自定义异常处理器已注册（BusinessException 处理器）
        assert utils.exception.BusinessException in app.exception_handlers
