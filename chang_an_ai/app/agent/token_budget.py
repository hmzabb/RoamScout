"""Token 预算控制（第四层循环控制层）：最大轮数 + 预算防死循环。

设计要点：
1. Token 计数：实时统计每轮和累计的 token 消耗
2. 预算上限：单会话总预算 + 单轮预算双重限制
3. 预警机制：达到 80% 预算时发出警告，防止突发超限
4. 优雅降级：超预算时给出友好提示而非直接报错

统计维度：
- input_tokens: 输入 token 数（用户消息 + 历史 + system prompt）
- output_tokens: 输出 token 数（LLM 回复 + tool_calls）
- 总消耗 = input_tokens + output_tokens

使用方式：
    from app.agent.token_budget import TokenBudget, BudgetExceededError

    budget = TokenBudget(session_id="sess_123")

    # 每轮调用前检查
    budget.check_before_call()

    # 每轮调用后记录
    budget.record_usage(input_tokens=1500, output_tokens=800)

    # 检查是否超预算
    if budget.is_exceeded():
        raise BudgetExceededError(...)
"""
from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BudgetStatus:
    """预算状态快照。"""
    session_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    current_round: int = 0
    max_rounds: int = 6
    budget_limit: int = 16000
    round_limit: int = 4000
    is_exceeded: bool = False
    is_warning: bool = False
    exceeded_reason: str = ""
    usage_ratio: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.budget_limit - self.total_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "remaining_tokens": self.remaining_tokens,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "usage_ratio": f"{self.usage_ratio:.1%}",
            "is_exceeded": self.is_exceeded,
            "is_warning": self.is_warning,
            "exceeded_reason": self.exceeded_reason or "",
        }


class BudgetExceededError(Exception):
    """预算超限异常。"""
    def __init__(self, status: BudgetStatus):
        self.status = status
        super().__init__(
            f"Token 预算超限: {status.exceeded_reason}, "
            f"已消耗 {status.total_tokens}/{status.budget_limit} "
            f"({status.usage_ratio:.1%})"
        )


class TokenBudget:
    """Token 预算管理器。

    功能：
    1. 会话级 token 预算追踪
    2. 单轮 token 预算检查
    3. 预警机制（80% 预算时告警）
    4. 预算超限时抛出异常
    5. 提供友好的预算使用摘要

    使用方式：
        budget = TokenBudget(session_id="sess_123")

        # LLM调用前
        budget.check_before_call(round_num=3)

        # LLM调用后
        budget.record_usage(
            input_tokens=1500,
            output_tokens=800,
            round_num=3,
        )

        # 获取状态
        status = budget.get_status()
    """

    def __init__(
        self,
        session_id: str,
        total_limit: int = None,
        round_limit: int = None,
        max_rounds: int = None,
    ):
        """
        Args:
            session_id: 会话ID
            total_limit: 单会话总 token 上限
            round_limit: 单轮 token 上限
            max_rounds: 最大轮数
        """
        self.session_id = session_id
        self._total_limit = total_limit or settings.max_total_tokens_per_session
        self._round_limit = round_limit or settings.max_tokens_per_round
        self._max_rounds = max_rounds or 6

        self._stats: Dict[str, Any] = {
            "rounds": {},  # round_num → {"input": ..., "output": ...}
            "created_at": time.time(),
        }

        # 每轮token记录
        self._total_input: int = 0
        self._total_output: int = 0
        self._current_round: int = 0
        self._is_exceeded: bool = False
        self._exceeded_reason: str = ""

    def check_before_call(self, round_num: int = 0) -> None:
        """LLM调用前的预算检查。

        Args:
            round_num: 当前轮数

        Raises:
            BudgetExceededError: 预算超限
        """
        self._current_round = round_num

        # 检查轮数限制
        if round_num > self._max_rounds:
            self._is_exceeded = True
            self._exceeded_reason = f"已达最大轮数 ({self._max_rounds})"
            raise BudgetExceededError(self.get_status())

        # 检查总预算
        if self._total_input + self._total_output >= self._total_limit:
            self._is_exceeded = True
            self._exceeded_reason = f"总 token 已达上限 ({self._total_limit})"
            raise BudgetExceededError(self.get_status())

        # 预警
        usage_ratio = (self._total_input + self._total_output) / self._total_limit
        if usage_ratio >= settings.warning_token_ratio:
            logger.warning(
                f"[TokenBudget] ⚠️ 预算预警: session={self.session_id[:8]}..., "
                f"已使用 {usage_ratio:.1%} ({self._total_input + self._total_output}/{self._total_limit})"
            )

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        round_num: int = 0,
    ) -> None:
        """记录一次 LLM 调用的 token 消耗。

        Args:
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数（含 tool_calls 的 JSON token）
            round_num: 当前轮数
        """
        # 检查单轮预算
        round_total = input_tokens + output_tokens
        if round_total > self._round_limit:
            logger.warning(
                f"[TokenBudget] 单轮 token ({round_total}) 超过限制 ({self._round_limit}), "
                f"session={self.session_id[:8]}..., round={round_num}"
            )

        self._total_input += input_tokens
        self._total_output += output_tokens

        # 记录每轮详情
        self._stats["rounds"][str(round_num)] = {
            "input": input_tokens,
            "output": output_tokens,
            "total": round_total,
            "cumulative": self._total_input + self._total_output,
        }

        total = self._total_input + self._total_output
        usage = total / self._total_limit * 100 if self._total_limit > 0 else 0

        logger.debug(
            f"[TokenBudget] 记录: session={self.session_id[:8]}..., "
            f"round={round_num}, "
            f"本轮 in={input_tokens}/out={output_tokens}, "
            f"累计 {total}/{self._total_limit} ({usage:.1f}%)"
        )

        # 检查是否超限
        if total >= self._total_limit:
            self._is_exceeded = True
            self._exceeded_reason = f"Token 预算超限: {total}/{self._total_limit}"

    def estimate_remaining_rounds(self) -> Tuple[int, str]:
        """估算剩余可用轮数。

        基于平均每轮消耗来估算还剩多少轮可以执行。

        Returns:
            (剩余轮数, 描述信息)
        """
        if self._current_round == 0:
            return (self._max_rounds, "尚未开始")

        total = self._total_input + self._total_output
        if total == 0:
            return (self._max_rounds - self._current_round, "暂无消耗")

        avg_per_round = total / self._current_round
        remaining_tokens = self._total_limit - total
        remaining_by_budget = int(remaining_tokens / avg_per_round) if avg_per_round > 0 else 99
        remaining_by_rounds = self._max_rounds - self._current_round
        remaining = min(remaining_by_budget, remaining_by_rounds)

        return (remaining, f"按预算约 {remaining_by_budget} 轮，按轮数限制 {remaining_by_rounds} 轮")

    def is_exceeded(self) -> bool:
        return self._is_exceeded

    def get_status(self) -> BudgetStatus:
        total = self._total_input + self._total_output
        return BudgetStatus(
            session_id=self.session_id,
            total_input_tokens=self._total_input,
            total_output_tokens=self._total_output,
            current_round=self._current_round,
            max_rounds=self._max_rounds,
            budget_limit=self._total_limit,
            round_limit=self._round_limit,
            is_exceeded=self._is_exceeded,
            is_warning=total / self._total_limit >= settings.warning_token_ratio if self._total_limit > 0 else False,
            exceeded_reason=self._exceeded_reason,
            usage_ratio=total / self._total_limit if self._total_limit > 0 else 0.0,
        )

    def get_budget_summary(self) -> str:
        """生成预算使用摘要（用于前端展示/日志）。"""
        status = self.get_status()
        remaining, desc = self.estimate_remaining_rounds()

        if status.is_exceeded:
            return f"⛔ 预算已超限: {status.exceeded_reason}"

        if status.is_warning:
            return (
                f"⚠️ 预算告警 | "
                f"已用: {status.total_tokens}/{status.budget_limit} ({status.usage_ratio:.0%}) | "
                f"剩余约 {remaining} 轮 | "
                f"当前第 {status.current_round}/{status.max_rounds} 轮"
            )

        return (
            f"💰 预算正常 | "
            f"已用: {status.total_tokens}/{status.budget_limit} ({status.usage_ratio:.0%}) | "
            f"剩余约 {remaining} 轮 | "
            f"当前第 {status.current_round}/{status.max_rounds} 轮"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """简易 token 数估算（中文约 1.5 字符/token，英文约 4 字符/token）。

        用于在没有 API 返回 usage 数据时的粗略估算。
        精确值建议使用 tiktoken 库。

        Args:
            text: 文本内容

        Returns:
            估算的 token 数
        """
        if not text:
            return 0
        # 简易估算：混合中英文
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


# 会话级别的预算管理器缓存
_budgets: Dict[str, TokenBudget] = {}


def get_session_budget(session_id: str, **kwargs) -> TokenBudget:
    """获取或创建会话级别的 TokenBudget。

    Args:
        session_id: 会话ID
        **kwargs: 传递给 TokenBudget 构造函数的参数

    Returns:
        TokenBudget 实例
    """
    if session_id not in _budgets:
        _budgets[session_id] = TokenBudget(session_id=session_id, **kwargs)
        logger.debug(f"[TokenBudget] 创建新预算: session={session_id[:8]}...")

    return _budgets[session_id]


def cleanup_session_budget(session_id: str) -> None:
    """清理会话的预算数据（会话过期时调用）。"""
    if session_id in _budgets:
        del _budgets[session_id]
        logger.debug(f"[TokenBudget] 清理预算: session={session_id[:8]}...")


def get_all_budget_stats() -> Dict[str, Any]:
    """获取所有活跃会话的预算统计（监控用）。"""
    stats = {}
    for sid, budget in _budgets.items():
        status = budget.get_status()
        stats[sid[:8]] = {
            "total_tokens": status.total_tokens,
            "usage_ratio": status.usage_ratio,
            "current_round": status.current_round,
            "is_warning": status.is_warning,
            "is_exceeded": status.is_exceeded,
        }
    return {"active_sessions": len(_budgets), "sessions": stats}