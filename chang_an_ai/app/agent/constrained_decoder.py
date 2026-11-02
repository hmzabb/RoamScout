"""约束解码（第二层·防护一）：用 JSON Schema 编译语法规则，拦截非法输出。

概念澄清——"约束解码"（Grammar-Constrained Decoding / Guided Decoding）指的是
在模型**生成阶段**做 token 级 logits 屏蔽：把 JSON Schema 编译成有限状态机/文法，
每一步只允许语法上合法的 token 参与采样，非法 token 概率置为 -inf。这样输出在
语法层面就是 100% 合法的，不存在"生成完再校验"的概率问题。

它必须能访问 logits，因此只有三种真实落地路径：

| 路径 | 代表实现 | 配置 guided_decoding_backend |
|------|----------|------------------------------|
| 自托管推理引擎 | vLLM `guided_json`、SGLang `json_schema` | "vllm" |
| 本地 GGUF 推理 | llama.cpp GBNF 文法 | "llamacpp" |
| 云 API | OpenAI Structured Outputs（strict）、DeepSeek JSON Output | "api" |

本项目当前使用 DeepSeek 云 API（OpenAI 兼容协议），云端不暴露 logits，
无法做 token 级屏蔽。因此本项目采用**等价的响应侧约束**组合来逼近同样的效果：

1. 请求侧：把工具的 JSON Schema 严格化（additionalProperties=false、
   全部属性 required、枚举封闭），缩小模型"合法输出空间"；
2. 响应侧：语法解析 → 失败则走 JSON 修复器（去 markdown 围栏/补括号/
   去尾逗号/单引号转双引号/截断补齐）→ 仍失败则结构化报错并触发重试。

与 `param_validator` 的分工：
- 本模块只管**语法/结构**是否合法（能不能解析、字段名对不对、类型对不对）
- `param_validator` 管**值域语义**（枚举、范围、长度、业务白名单）

使用方式：
    from app.agent.constrained_decoder import get_constrained_decoder

    decoder = get_constrained_decoder()
    result = decoder.decode(raw_content)
    if not result.ok:
        # 把 result.error 回传给模型重试
        ...
    tool_calls = result.data["tool_calls"]
"""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    """约束解码结果。"""

    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    repaired: bool = False
    stage: str = "raw"
    repair_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "repaired": self.repaired,
            "stage": self.stage,
            "error": self.error,
            "repair_log": self.repair_log,
        }


class GrammarConstrainedDecoder:
    """JSON 语法约束解码器。

    职责：
    1. 把 LangChain 工具的 input schema 严格化为"封闭 Schema"
    2. 解析模型输出（含 markdown 围栏、前后自然语言、语法错误）
    3. 修复失败时产出结构化错误，供上层回传模型重试

    真正的 token 级屏蔽在 vLLM/llama.cpp 后端生效（backend != "api"），
    本类在 "api" 后端下退化为"解析 + 修复 + 校验"的确定性流程。
    """

    # 常见语法修复规则（按顺序应用）
    _FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
    _TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
    _SMART_QUOTE_RE = re.compile(r"[\u201c\u201d]")

    def __init__(self, backend: Optional[str] = None, max_repair: Optional[int] = None):
        self.backend = backend or getattr(settings, "guided_decoding_backend", "api")
        self.max_repair = (
            max_repair
            if max_repair is not None
            else getattr(settings, "constrained_decode_max_repair", 1)
        )

    @property
    def is_token_level(self) -> bool:
        """是否具备真正的 token 级屏蔽能力（自托管引擎才为 True）。"""
        return self.backend in ("vllm", "llamacpp", "sglang")

    def strictify_schema(
        self,
        schema: Dict[str, Any],
        force_all_required: bool = False,
    ) -> Dict[str, Any]:
        """把工具 Schema 严格化：封闭对象、禁用未声明字段。

        这是请求侧的"空间压缩"：模型可选的字段越少，越不容易生成非法结构。

        注意 required 的处理：
        - 默认（force_all_required=False）保留原 required，可选参数仍是可选。
          因为本项目工具存在可选参数（如 list_shops_by_type 的 x/y），
          若强行全量必填，模型会被迫为"没有值"的可选参数编造值。
        - force_all_required=True 时对齐 OpenAI Structured Outputs 的严格模式
          （required 必须覆盖全部属性），仅在 guided_decoding_backend="api"
          且服务端确实支持 strict 时才使用。

        Args:
            schema: 工具原始 input schema
            force_all_required: 是否强制 required 覆盖全部属性

        Returns:
            严格化后的 schema（不修改原对象）
        """
        if not isinstance(schema, dict):
            return schema

        result = json.loads(json.dumps(schema, ensure_ascii=False))

        def _walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            if node.get("type") == "object" and "properties" in node:
                node.setdefault("additionalProperties", False)
                props = node["properties"]
                if isinstance(props, dict):
                    if force_all_required:
                        node["required"] = list(props.keys())
                    for sub in props.values():
                        _walk(sub)
            items = node.get("items")
            if isinstance(items, dict):
                _walk(items)
            for key in ("anyOf", "oneOf", "allOf"):
                if isinstance(node.get(key), list):
                    for sub in node[key]:
                        _walk(sub)

        _walk(result)
        return result

    def decode(self, raw_text: str) -> DecodeResult:
        """解析模型输出为结构化 JSON。

        Args:
            raw_text: 模型返回的原始文本（可能带 markdown 围栏或解释文字）

        Returns:
            DecodeResult，ok=True 时 data 为解析出的 dict
        """
        if not raw_text or not raw_text.strip():
            return DecodeResult(ok=False, error="模型输出为空", stage="empty")

        # 第 1 阶段：直接解析（正常路径，云 API + tools 参数下命中率最高）
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                return DecodeResult(ok=True, data=data, stage="direct")
        except json.JSONDecodeError:
            pass

        # 第 2 阶段：语法修复（最多 max_repair 轮）
        repaired_text, log = self._repair(raw_text)
        for attempt in range(max(1, self.max_repair)):
            try:
                data = json.loads(repaired_text)
                if isinstance(data, dict):
                    logger.info(
                        f"[ConstrainedDecode] JSON 修复成功(attempt={attempt + 1}): {log}"
                    )
                    return DecodeResult(
                        ok=True, data=data, repaired=True,
                        stage="repaired", repair_log=log,
                    )
            except json.JSONDecodeError as e:
                log.append(f"第{attempt + 1}轮仍失败: {e.msg} (pos={e.pos})")
                repaired_text, more = self._repair(repaired_text)
                log.extend(more)

        # 第 3 阶段：结构化报错，交给上层回传模型重试
        error = (
            "❌ 输出不是合法 JSON，无法解析为工具调用。\n"
            f"修复过程: {'; '.join(log[-3:])}\n"
            "请严格按 JSON 格式输出，不要添加解释文字或 markdown 围栏。"
        )
        logger.warning(f"[ConstrainedDecode] 解析失败: {log}")
        return DecodeResult(ok=False, error=error, stage="failed", repair_log=log)

    def _repair(self, text: str) -> tuple[str, List[str]]:
        """对常见 JSON 语法错误做确定性修复。

        Args:
            text: 待修复文本

        Returns:
            (修复后文本, 修复动作日志)
        """
        log: List[str] = []
        s = text.strip()

        # 1. 去掉 markdown 围栏
        if "```" in s:
            s = self._FENCE_RE.sub("", s)
            if "```" in s:
                s = re.sub(r"```[a-zA-Z]*", "", s)
            log.append("移除markdown围栏")

        # 2. 中文全角引号 → 半角
        if self._SMART_QUOTE_RE.search(s):
            s = self._SMART_QUOTE_RE.sub('"', s)
            log.append("全角引号转半角")

        # 3. 截取最外层 JSON 片段（去掉前后解释文字）
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            if start > 0 or end < len(s) - 1:
                s = s[start:end + 1]
                log.append("截取最外层JSON片段")
        elif start != -1 and end == -1:
            # 被截断：只有左括号
            s = s[start:]
            log.append("检测到截断输出")

        # 4. 去尾逗号
        new_s = self._TRAILING_COMMA_RE.sub(r"\1", s)
        if new_s != s:
            s = new_s
            log.append("移除尾逗号")

        # 5. 补全未闭合的括号/引号
        s, fixed = self._balance_brackets(s)
        if fixed:
            log.append("补全未闭合括号")

        # 6. Python 字面量兜底（None/True/False/单引号）
        if "None" in s or "True" in s or "False" in s:
            s = s.replace("None", "null").replace("True", "true").replace("False", "false")
            log.append("Python字面量转JSON")

        if not log:
            log.append("无需修复")
        return s, log

    @staticmethod
    def _balance_brackets(s: str) -> tuple[str, bool]:
        """补全未闭合的括号，并去掉末尾多余逗号。"""
        stack: List[str] = []
        in_string = False
        escape = False
        pairs = {"}": "{", "]": "["}

        for ch in s:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in pairs:
                if stack and stack[-1] == pairs[ch]:
                    stack.pop()

        if in_string:
            s += '"'
        if not stack:
            return s, in_string

        s = s.rstrip().rstrip(",")
        for opener in reversed(stack):
            s += "}" if opener == "{" else "]"
        return s, True

    def extract_tool_calls(self, raw_text: str) -> DecodeResult:
        """从纯文本输出中抽取 tool_calls（部分模型不返回结构化字段）。

        兼容两种格式：
        1. {"tool_calls": [{"name": ..., "args": ...}]}
        2. {"name": ..., "arguments": {...}}  ← OpenAI 风格

        Args:
            raw_text: 模型原始输出

        Returns:
            DecodeResult，data 为归一化后的 {"tool_calls": [...]}
        """
        result = self.decode(raw_text)
        if not result.ok or not result.data:
            return result

        data = result.data
        raw_calls = data.get("tool_calls")
        if raw_calls is None and "name" in data:
            raw_calls = [data]

        if not isinstance(raw_calls, list):
            return DecodeResult(
                ok=False,
                error="❌ JSON 中缺少 tool_calls 字段",
                stage="schema",
            )

        normalized: List[Dict[str, Any]] = []
        for idx, call in enumerate(raw_calls):
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("tool") or call.get("function", {}).get("name")
            args = call.get("args") or call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    try:
                        args = ast.literal_eval(args)
                    except (ValueError, SyntaxError):
                        args = {}
            if not name:
                continue
            normalized.append({"name": name, "args": args or {}, "id": call.get("id") or f"call_{idx}"})

        if not normalized:
            return DecodeResult(ok=False, error="❌ tool_calls 为空或结构不合法", stage="schema")

        return DecodeResult(
            ok=True,
            data={"tool_calls": normalized},
            repaired=result.repaired,
            stage=result.stage,
            repair_log=result.repair_log,
        )


_decoder: Optional[GrammarConstrainedDecoder] = None


def get_constrained_decoder() -> GrammarConstrainedDecoder:
    """工厂：全局单例约束解码器。"""
    global _decoder
    if _decoder is None:
        _decoder = GrammarConstrainedDecoder()
    return _decoder