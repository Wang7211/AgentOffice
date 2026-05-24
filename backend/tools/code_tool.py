"""安全算术执行工具。"""

import ast
import math
import operator
import re
from typing import Any
from typing import Callable

from tools.base import BaseTool
from tools.base import ToolResult
from utils.exception import ToolException


BinaryOperator = Callable[[float, float], float]
UnaryOperator = Callable[[float], float]


class SafeExpressionEvaluator:
    """在不使用动态执行的情况下计算算术表达式。"""

    _binary_operators: dict[type[ast.AST], BinaryOperator] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operators: dict[type[ast.AST], UnaryOperator] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _functions: dict[str, Callable[..., float]] = {
        "abs": abs,
        "ceil": math.ceil,
        "floor": math.floor,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "sum": sum,
        "max": max,
        "min": min,
    }
    _constants: dict[str, float] = {"pi": math.pi, "e": math.e}

    def evaluate(self, expression: str) -> float:
        """计算安全算术表达式。

        参数:
            expression: 算术表达式。

        返回:
            数值计算结果。

        异常:
            ToolException: 表达式包含不安全语法时抛出。
        """
        try:
            expression_ast = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolException("表达式语法错误") from exc
        return float(self._eval_node(expression_ast.body))

    def _eval_node(self, node: ast.AST) -> float:
        """递归计算 AST 节点。"""
        if isinstance(node, ast.Constant):
            return self._eval_constant(node)
        if isinstance(node, ast.BinOp):
            return self._eval_binary(node)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unary(node)
        if isinstance(node, ast.Name):
            return self._eval_name(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.List | ast.Tuple):
            raise ToolException("列表只能作为函数参数使用")
        raise ToolException("表达式包含不允许的语法")

    def _eval_constant(self, node: ast.Constant) -> float:
        """计算数字常量。"""
        if not isinstance(node.value, int | float):
            raise ToolException("表达式仅支持数字")
        return float(node.value)

    def _eval_binary(self, node: ast.BinOp) -> float:
        """计算二元表达式。"""
        operator_func = self._binary_operators.get(type(node.op))
        if not operator_func:
            raise ToolException("表达式包含不支持的运算符")
        return operator_func(self._eval_node(node.left), self._eval_node(node.right))

    def _eval_unary(self, node: ast.UnaryOp) -> float:
        """计算一元表达式。"""
        operator_func = self._unary_operators.get(type(node.op))
        if not operator_func:
            raise ToolException("表达式包含不支持的一元运算符")
        return operator_func(self._eval_node(node.operand))

    def _eval_name(self, node: ast.Name) -> float:
        """计算白名单常量。"""
        if node.id not in self._constants:
            raise ToolException(f"不允许的变量：{node.id}")
        return self._constants[node.id]

    def _eval_call(self, node: ast.Call) -> float:
        """计算白名单函数调用。"""
        if not isinstance(node.func, ast.Name):
            raise ToolException("函数调用格式不合法")
        func = self._functions.get(node.func.id)
        if not func:
            raise ToolException(f"不允许的函数：{node.func.id}")
        args = [self._eval_argument(argument) for argument in node.args]
        return float(func(*args))

    def _eval_argument(self, node: ast.AST) -> float | list[float]:
        """计算函数参数。"""
        if isinstance(node, ast.List | ast.Tuple):
            return [self._eval_node(item) for item in node.elts]
        return self._eval_node(node)


class CodeTool(BaseTool):
    """面向办公数值任务的安全计算工具。"""

    name = "code"
    description = "执行安全数学表达式，支持加减乘除、幂、常用数学函数。"
    input_schema = {"expression": "必填，数学表达式，例如 sqrt(16)+2*3。"}

    def __init__(self) -> None:
        self._evaluator = SafeExpressionEvaluator()

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """执行安全算术表达式。

        参数:
            tool_input: 包含 `expression` 的字典。

        返回:
            计算结果。

        异常:
            ToolException: 表达式为空或不安全时抛出。
        """
        expression = str(tool_input.get("expression", "")).strip()
        expression = self._clean_expression(expression)
        if not expression:
            raise ToolException("计算表达式不能为空")
        result = self._evaluator.evaluate(expression)
        return ToolResult(
            content=f"计算结果：{result:g}",
            metadata={"expression": expression, "result": result},
        )

    def _clean_expression(self, expression: str) -> str:
        """移除表达式中常见的自然语言包装。"""
        cleaned_expression = expression.replace("×", "*").replace("÷", "/")
        cleaned_expression = re.sub(
            r"^(计算|算一下|帮我算|请计算)[:：]?",
            "",
            cleaned_expression,
        )
        cleaned_expression = re.sub(
            r"[^0-9A-Za-z_+\-*/().,\[\]\s]",
            "",
            cleaned_expression,
        )
        return cleaned_expression.strip()
