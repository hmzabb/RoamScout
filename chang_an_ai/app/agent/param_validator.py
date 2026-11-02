"""参数校验（第二层·防护二）：执行前用 JSON Schema 编译规则校验参数。

与 `constrained_decoder` 的分工：
- constrained_decoder：保证"能不能解析出来"（语法层）
- param_validator：保证"解析出来的值对不对"（语义层）

校验维度（对应 JSON Schema 关键字）：
| 维度 | Schema 关键字 | 典型错误 |
|------|---------------|----------|
| 必填 | required | 模型漏传 type_name |
| 类型 | type | 数字传成字符串、数组传成对象 |
| 枚举 | enum | type_name 传了"火锅"（合法值是"美食"） |
| 范围 | minimum / maximum | 坐标越界、limit 为负数 |
| 长度 | minLength / maxLength | 搜索词过长 |
| 正则 | pattern | 格式不符 |

关键设计——**报错要能驱动模型自我修正**：
校验失败不是直接抛异常终止，而是把结构化的真实报错（哪个工具、哪个参数、
期望什么、实际什么、可选值有哪些）作为 ToolMessage 回传给模型，让它在
下一轮重新生成合法调用。这才是图中"校验失败把真实报错回传给模型修正"。

不依赖第三方 jsonschema 库（项目 requirements 未包含），本模块自行实现
JSON Schema 子集校验，覆盖本项目 7 个工具用到的全部关键字。

使用方式：
    from app.agent.param_validator import get_param_validator

    validator = get_param_validator()
    errors = validator.validate(tool_name, args, schema)
    if errors:
        msg = validator.format_for_llm(errors)   # 回传给模型
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ParamError:
    """单个参数的校验错误（结构化，便于回传给模型和写入审计）。"""

    tool_name: str
    param: str
    code: str
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    allowed: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "param": self.param,
            "code": self.code,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "allowed": self.allowed,
        }

    def to_llm_line(self, index: int) -> str:
        """转成模型可读的一行提示。"""
        line = f"{index}. [{self.code}] 参数「{self.param}」{self.message}"
        if self.expected:
            line += f"（期望: {self.expected}）"
        if self.allowed:
            preview = self.allowed[:8]
            line += f"（可选值: {preview}{'...' if len(self.allowed) > 8 else ''}）"
        return line


# JSON Schema type → Python 类型判定
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


class ParamValidator:
    """JSON Schema 子集校验器。

    校验顺序：必填 → 未知字段 → 类型 → 枚举 → 范围 → 长度 → 正则 → 嵌套
    """

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[ParamError]:
        """校验工具参数是否满足其 JSON Schema。

        Args:
            tool_name: 工具名
            args: 模型给出的参数
            schema: 工具的 input schema

        Returns:
            ParamError 列表（空列表表示全部通过）
        """
        if not getattr(settings, "param_validation_enabled", True):
            return []

        if not isinstance(args, dict):
            return [ParamError(
                tool_name=tool_name, param="<args>", code="ARGS_NOT_OBJECT",
                message="参数整体不是对象", actual=str(type(args)),
            )]

        errors: List[ParamError] = []
        properties: Dict[str, Any] = schema.get("properties", {}) or {}
        required: List[str] = schema.get("required", []) or []

        # 1. 必填校验
        for param in required:
            if param not in args or args[param] is None:
                prop = self._resolve_prop(properties.get(param, {}))
                errors.append(ParamError(
                    tool_name=tool_name, param=param, code="MISSING_REQUIRED",
                    message="缺少必填参数，必须提供",
                    expected=prop.get("type", "any"),
                    allowed=prop.get("enum"),
                ))

        # 2. 未知字段校验（严格模式下拒绝模型臆造的参数）
        if schema.get("additionalProperties") is False:
            for param in args:
                if param.startswith("_"):  # 内部标记字段（如 _confirm_token）放行
                    continue
                if param not in properties:
                    errors.append(ParamError(
                        tool_name=tool_name, param=param, code="UNKNOWN_PARAM",
                        message=f"不是该工具声明的参数，可用参数: {list(properties.keys())}",
                        actual=str(param),
                    ))

        # 3. 逐字段校验
        for param, prop in properties.items():
            if param not in args or args[param] is None:
                continue
            errors.extend(self._validate_value(tool_name, param, args[param], prop))

        return errors

    def validate_and_normalize(
        self,
        tool_name: str,
        args: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[List[ParamError], Dict[str, Any]]:
        """校验并把"类型看起来对但写法不对"的值规范化后返回。

        为什么需要：模型常把数字写成字符串（"108.94"）。若只做宽容校验而不回写，
        校验虽然通过，但工具执行时拿到的仍是字符串，会在数值比较/计算处出错。
        因此这里返回规范化后的参数副本，调用方应替换掉原始 args。

        Args:
            tool_name: 工具名
            args: 模型给出的参数
            schema: 工具的 input schema

        Returns:
            (错误列表, 规范化后的参数)
        """
        if not isinstance(args, dict):
            return self.validate(tool_name, args, schema), args if isinstance(args, dict) else {}

        properties: Dict[str, Any] = schema.get("properties", {}) or {}
        normalized: Dict[str, Any] = dict(args)
        for param, prop in properties.items():
            if param not in normalized or normalized[param] is None:
                continue
            if isinstance(prop, dict):
                normalized[param] = self._normalize_value(normalized[param], prop)

        return self.validate(tool_name, normalized, schema), normalized

    @staticmethod
    def _resolve_prop(prop: Dict[str, Any]) -> Dict[str, Any]:
        """把 anyOf/oneOf 包裹的可选类型展开成直接类型。

        为什么需要：Pydantic 为 `Optional[float]` 生成的是
        `{"anyOf": [{"type": "number"}, {"type": "null"}]}`，
        顶层没有 type 关键字。若不展开，类型/枚举/范围校验会被整体跳过，
        导致"看起来在校验、实际没校验"。

        Args:
            prop: 字段 Schema 片段

        Returns:
            展开后的 Schema（外层关键字优先于分支内的同名关键字）
        """
        if not isinstance(prop, dict):
            return {}

        for key in ("anyOf", "oneOf"):
            branches = prop.get(key)
            if not isinstance(branches, list):
                continue
            non_null = [
                b for b in branches
                if isinstance(b, dict) and b.get("type") not in (None, "null")
            ]
            if len(non_null) == 1:
                merged = dict(non_null[0])
                for k, v in prop.items():
                    if k not in ("anyOf", "oneOf"):
                        merged[k] = v
                return merged

        return prop

    def _normalize_value(self, value: Any, prop: Dict[str, Any]) -> Any:
        """按 Schema 递归规范化单个值（仅做无损类型转换）。

        Args:
            value: 原始值
            prop: 该字段的 Schema 片段

        Returns:
            规范化后的值；无法转换时原样返回（留给校验阶段报错）
        """
        prop = self._resolve_prop(prop)
        expected_type = prop.get("type")
        if expected_type and not self._check_type(value, expected_type):
            coerced = self._try_coerce(value, expected_type)
            if coerced is not _COERCE_FAILED:
                value = coerced

        items = prop.get("items")
        if isinstance(value, list) and isinstance(items, dict):
            value = [self._normalize_value(item, items) for item in value]

        return value

    def _validate_value(
        self,
        tool_name: str,
        param: str,
        value: Any,
        prop: Dict[str, Any],
    ) -> List[ParamError]:
        """校验单个值（递归处理 array/object 嵌套）。"""
        errors: List[ParamError] = []
        prop = self._resolve_prop(prop)
        expected_type = prop.get("type")

        # 类型（含隐式转换尝试：模型常把数字写成字符串）
        if expected_type:
            if not self._check_type(value, expected_type):
                coerced = self._try_coerce(value, expected_type)
                if coerced is not _COERCE_FAILED:
                    logger.debug(
                        f"[ParamValidator] 参数「{param}」自动转换: "
                        f"{type(value).__name__} → {expected_type}"
                    )
                    value = coerced
                else:
                    errors.append(ParamError(
                        tool_name=tool_name, param=param, code="TYPE_MISMATCH",
                        message=f"类型不匹配，实际为 {type(value).__name__}",
                        expected=expected_type, actual=str(value)[:50],
                    ))
                    return errors

        # 枚举
        enum_values = prop.get("enum")
        if enum_values and value not in enum_values:
            suggestion = self._closest(value, enum_values)
            hint = f"，是否想用「{suggestion}」？" if suggestion else ""
            errors.append(ParamError(
                tool_name=tool_name, param=param, code="ENUM_MISMATCH",
                message=f"值「{value}」不在允许集合中{hint}",
                actual=str(value), allowed=list(enum_values),
            ))

        # 数值范围
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in prop and value < prop["minimum"]:
                errors.append(ParamError(
                    tool_name=tool_name, param=param, code="OUT_OF_RANGE",
                    message=f"值 {value} 小于下限 {prop['minimum']}",
                    expected=f">= {prop['minimum']}", actual=str(value),
                ))
            if "maximum" in prop and value > prop["maximum"]:
                errors.append(ParamError(
                    tool_name=tool_name, param=param, code="OUT_OF_RANGE",
                    message=f"值 {value} 大于上限 {prop['maximum']}",
                    expected=f"<= {prop['maximum']}", actual=str(value),
                ))

        # 字符串长度与正则
        if isinstance(value, str):
            if "minLength" in prop and len(value) < prop["minLength"]:
                errors.append(ParamError(
                    tool_name=tool_name, param=param, code="TOO_SHORT",
                    message=f"长度 {len(value)} 小于下限 {prop['minLength']}",
                    expected=f"长度 >= {prop['minLength']}", actual=value[:50],
                ))
            if "maxLength" in prop and len(value) > prop["maxLength"]:
                errors.append(ParamError(
                    tool_name=tool_name, param=param, code="TOO_LONG",
                    message=f"长度 {len(value)} 超过上限 {prop['maxLength']}",
                    expected=f"长度 <= {prop['maxLength']}", actual=value[:50],
                ))
            if "pattern" in prop:
                try:
                    if not re.search(prop["pattern"], value):
                        errors.append(ParamError(
                            tool_name=tool_name, param=param, code="PATTERN_MISMATCH",
                            message=f"不匹配格式 {prop['pattern']}",
                            expected=prop["pattern"], actual=value[:50],
                        ))
                except re.error:
                    logger.warning(f"[ParamValidator] 非法正则: {prop['pattern']}")

        # 数组元素递归
        if isinstance(value, list) and isinstance(prop.get("items"), dict):
            for idx, item in enumerate(value):
                errors.extend(self._validate_value(tool_name, f"{param}[{idx}]", item, prop["items"]))

        return errors

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        check = _TYPE_CHECKS.get(expected)
        if check is None:
            return True
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return check(value)

    @staticmethod
    def _try_coerce(value: Any, expected: str) -> Any:
        """尝试把模型给的"看起来对但类型不对"的值转成目标类型。"""
        try:
            if expected == "string" and isinstance(value, (int, float)):
                return str(value)
            if expected == "integer" and isinstance(value, str):
                return int(float(value.strip()))
            if expected == "number" and isinstance(value, str):
                return float(value.strip())
            if expected == "boolean" and isinstance(value, str):
                if value.strip().lower() in ("true", "1", "yes"):
                    return True
                if value.strip().lower() in ("false", "0", "no"):
                    return False
            if expected == "array" and isinstance(value, str):
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
        except (ValueError, TypeError, json.JSONDecodeError):
            return _COERCE_FAILED
        return _COERCE_FAILED

    @staticmethod
    def _closest(value: Any, candidates: List[Any]) -> Optional[Any]:
        """在候选值里找最接近的一个（用于枚举错误提示）。"""
        if not candidates:
            return None
        target = str(value)
        best, best_score = None, 0.0
        for cand in candidates:
            c = str(cand)
            if target == c:
                return cand
            score = 0.0
            if target and c:
                if target in c or c in target:
                    score = 0.9
                else:
                    common = len(set(target) & set(c))
                    score = common / max(len(set(target)), len(set(c)))
            if score > best_score:
                best, best_score = cand, score
        return best if best_score >= 0.3 else None

    def format_for_llm(self, errors: List[ParamError]) -> str:
        """把结构化错误转成回传给模型的修正指令。

        这是"让模型自我修正"闭环的关键输入：真实报错 + 明确期望 + 可选值。

        Args:
            errors: 校验错误列表

        Returns:
            可直接作为 ToolMessage content 的文本
        """
        if not errors:
            return ""

        lines = ["❌ 工具参数校验失败，请修正后重新调用："]
        for i, err in enumerate(errors, 1):
            lines.append(err.to_llm_line(i))
        lines.append("请严格按工具的 JSON Schema 重新生成参数，不要臆造字段或枚举值。")
        return "\n".join(lines)


_COERCE_FAILED = object()

_validator: Optional[ParamValidator] = None


def get_param_validator() -> ParamValidator:
    """工厂：全局单例参数校验器。"""
    global _validator
    if _validator is None:
        _validator = ParamValidator()
    return _validator