"""SafeExpressionEvaluator 和 CodeTool 测试用例。"""

from typing import Any

import pytest

from tools.code_tool import CodeTool
from tools.code_tool import SafeExpressionEvaluator
from utils.exception import ToolException


# ---------------------------------------------------------------------------
# SafeExpressionEvaluator — 核心安全运算逻辑
# ---------------------------------------------------------------------------

class TestSafeExpressionEvaluator:
    def setup_method(self) -> None:
        self._evaluator = SafeExpressionEvaluator()

    # --- 基础运算 ---

    def test_addition(self) -> None:
        assert self._evaluator.evaluate("1 + 2") == 3.0

    def test_subtraction(self) -> None:
        assert self._evaluator.evaluate("10 - 4") == 6.0

    def test_multiplication(self) -> None:
        assert self._evaluator.evaluate("3 * 4") == 12.0

    def test_division(self) -> None:
        assert self._evaluator.evaluate("10 / 4") == 2.5

    def test_floor_div(self) -> None:
        assert self._evaluator.evaluate("10 // 3") == 3.0

    def test_modulo(self) -> None:
        assert self._evaluator.evaluate("10 % 3") == 1.0

    def test_power(self) -> None:
        assert self._evaluator.evaluate("2 ** 3") == 8.0

    # --- 一元运算 ---

    def test_unary_minus(self) -> None:
        assert self._evaluator.evaluate("-5") == -5.0

    def test_unary_plus(self) -> None:
        assert self._evaluator.evaluate("+3") == 3.0

    # --- 复合表达式 ---

    def test_complex_expression(self) -> None:
        result = self._evaluator.evaluate("(2 + 3) * 4")
        assert result == 20.0

    def test_chained_operations(self) -> None:
        result = self._evaluator.evaluate("2 + 3 * 4 - 1")
        assert result == 13.0  # 优先级: 乘法优先

    # --- 常量 ---

    def test_pi_constant(self) -> None:
        import math

        assert self._evaluator.evaluate("pi") == math.pi

    def test_e_constant(self) -> None:
        import math

        assert self._evaluator.evaluate("e") == math.e

    # --- 函数 ---

    def test_sqrt_function(self) -> None:
        assert self._evaluator.evaluate("sqrt(16)") == 4.0

    def test_abs_function(self) -> None:
        assert self._evaluator.evaluate("abs(-10)") == 10.0

    def test_sum_function(self) -> None:
        assert self._evaluator.evaluate("sum([1, 2, 3])") == 6.0

    def test_max_function(self) -> None:
        assert self._evaluator.evaluate("max(3, 7, 5)") == 7.0

    def test_nested_functions(self) -> None:
        assert round(self._evaluator.evaluate("sqrt(2)"), 3) == 1.414

    # --- 异常路径 ---

    def test_syntax_error_raises(self) -> None:
        with pytest.raises(ToolException, match="语法错误"):
            self._evaluator.evaluate("2 + * 3")

    def test_unknown_variable_raises(self) -> None:
        with pytest.raises(ToolException, match="不允许的变量"):
            self._evaluator.evaluate("x + 1")

    def test_unknown_function_raises(self) -> None:
        with pytest.raises(ToolException, match="不允许的函数"):
            self._evaluator.evaluate("foobar(42)")

    def test_string_constant_raises(self) -> None:
        with pytest.raises(ToolException, match="仅支持数字"):
            self._evaluator.evaluate("'hello'")

    def test_list_as_statement_raises(self) -> None:
        with pytest.raises(ToolException, match="列表只能作为函数参数"):
            self._evaluator.evaluate("[1, 2, 3]")

    def test_unsupported_operator_raises(self) -> None:
        with pytest.raises(ToolException, match="不支持的运算符"):
            from ast import AST, BinOp

            class _FakeOp(AST):
                pass

            self._evaluator._eval_binary(
                BinOp(left=None, op=_FakeOp(), right=None)  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# CodeTool — 集成 SafeExpressionEvaluator 的对外工具
# ---------------------------------------------------------------------------

class TestCodeTool:
    def setup_method(self) -> None:
        self._tool = CodeTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "code"
        assert "expression" in spec.input_schema

    def test_run_simple(self) -> None:
        result = self._tool.run({"expression": "2 + 2"})
        assert "4" in result.content

    def test_run_with_chinese_prefix(self) -> None:
        result = self._tool.run({"expression": "计算：3 * 5"})
        assert "15" in result.content

    def test_run_cleans_special_chars(self) -> None:
        result = self._tool.run({"expression": "帮我算 2 × 3 ÷ 4"})
        assert "1.5" in result.content

    def test_empty_expression_raises(self) -> None:
        with pytest.raises(ToolException, match="不能为空"):
            self._tool.run({"expression": ""})

    def test_metadata_contains_result(self) -> None:
        result = self._tool.run({"expression": "sqrt(144)"})
        assert result.metadata["result"] == 12.0
