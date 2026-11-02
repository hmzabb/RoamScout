"""Trace 系统（第五层可观测）：串联完整因果链。

设计要点：
- 每次请求生成唯一 trace_id，贯穿 Agent → 工具 → LLM 全链路
- 每个节点（路由/Agent思考/工具执行/LLM调用）生成独立 span
- Span 之间通过 parent_span_id 串联父子关系，还原完整调用树
- 支持采样率控制，避免海量日志
- Trace 数据写入 Redis（轻量级，复用已有基础设施）

事件类型：
- route: 路由决策
- agent_think: Agent 思考决策
- tool_call: 工具调用开始
- tool_result: 工具调用结束（含耗时、质量检查结果）
- llm_call: LLM API 调用
- budget_check: Token预算检查
- policy_check: 策略引擎检查
- error: 异常事件

使用方式：
    from app.agent.tracer import TraceContext, tracer

    with TraceContext(session_id="xxx", user_query="...") as ctx:
        # 所有子调用自动关联到 ctx.trace_id
        with tracer.span("agent_think") as span:
            span.set_tag("tools_called", ["list_shops_by_type"])
            ...
        with tracer.span("tool_call", parent=span) as tool_span:
            ...
"""
from __future__ import annotations

import json
import time
import uuid
import random
import logging
from contextvars import ContextVar
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta

from app.config import settings

logger = logging.getLogger(__name__)

# 线程安全的上下文变量，存储当前 trace context
_current_trace: ContextVar[Optional["TraceContext"]] = ContextVar("_current_trace", default=None)


@dataclass
class Span:
    """Trace 中的一个操作片段。

    每个 span 代表一个独立的操作节点（如一次工具调用、一次LLM请求）。
    通过 parent_span_id 串联父子关系，构建完整调用树。
    """

    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running / success / error
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    def set_tag(self, key: str, value: Any) -> None:
        """设置标签（记录关键信息）。"""
        self.tags[key] = value

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        """添加事件（记录时间点信息）。"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def mark_success(self) -> None:
        """标记成功完成。"""
        self.end_time = time.time()
        self.status = "success"

    def mark_error(self, error: str) -> None:
        """标记失败。"""
        self.end_time = time.time()
        self.status = "error"
        self.error_message = error

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）。"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于序列化）。"""
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "tags": self.tags,
            "events": self.events,
            "error": self.error_message,
        }


class TraceContext:
    """一次请求的完整 Trace 上下文。

    每个用户请求创建一个 TraceContext，包含：
    - 全局唯一 trace_id
    - 所有子 span 的集合
    - 请求元信息（session_id, user_query, 时间戳等）

    使用 with 语句自动管理生命周期：
        with TraceContext(session_id, user_query) as ctx:
            ...
        # 退出时自动持久化
    """

    __slots__ = (
        "trace_id", "session_id", "user_query", "start_time",
        "end_time", "spans", "metadata", "_persisted",
        "input_tokens", "output_tokens",
    )

    def __init__(self, session_id: str, user_query: str):
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)

        self.trace_id = f"trace_{now.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.user_query = user_query[:200]
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.spans: List[Span] = []
        # 单任务 token 成本（由 agent_node 每轮累加）
        self.input_tokens = 0
        self.output_tokens = 0
        self.metadata: Dict[str, Any] = {
            "session_id": session_id,
            "user_query": user_query[:200],
            "start_iso": now.isoformat(),
        }
        self._persisted = False

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """累加本次任务的 token 消耗（第五层：单任务 token 成本）。"""
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def __enter__(self) -> "TraceContext":
        _current_trace.set(self)
        logger.debug(f"[Trace] 开始追踪: trace_id={self.trace_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.metadata["end_iso"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        self.metadata["total_duration_ms"] = round((self.end_time - self.start_time) * 1000, 2)
        self.metadata["input_tokens"] = self.input_tokens
        self.metadata["output_tokens"] = self.output_tokens
        self.metadata["total_tokens"] = self.input_tokens + self.output_tokens

        if exc_type:
            self.metadata["error"] = str(exc_val)

        # 持久化到 Redis（异步写入，不阻塞）
        if settings.tracing_enabled and self._should_sample():
            self._persist_to_redis()

        _current_trace.set(None)
        logger.debug(
            f"[Trace] 追踪结束: trace_id={self.trace_id}, "
            f"spans={len(self.spans)}, "
            f"duration={self.metadata.get('total_duration_ms', 0)}ms"
        )

    def _should_sample(self) -> bool:
        """根据采样率决定是否记录此 trace。"""
        if settings.trace_sample_rate >= 1.0:
            return True
        return random.random() < settings.trace_sample_rate

    def _persist_to_redis(self) -> None:
        """将 trace 数据写入 Redis（非阻塞）。"""
        if self._persisted:
            return
        try:
            import redis as _redis
            r = _redis.Redis.from_url(
                settings.redis_url,
                password=settings.redis_password or None,
                socket_connect_timeout=1,
            )
            key = f"trace:{self.trace_id}"
            data = self.to_dict()
            r.setex(key, settings.trace_redis_ttl, json.dumps(data, ensure_ascii=False, default=str))
            self._persisted = True
            logger.debug(f"[Trace] 已持久化 trace_id={self.trace_id}")
        except Exception as e:
            logger.warning(f"[Trace] 持久化失败: {e}")

    def create_span(self, name: str, parent_span_id: str = None) -> Span:
        """创建新的子 span。

        Args:
            name: span 名称（如 "tool_call", "agent_think", "llm_call"）
            parent_span_id: 父 span 的 ID（可选）

        Returns:
            新创建的 Span 对象
        """
        span = Span(
            span_id=f"span_{uuid.uuid4().hex[:8]}",
            parent_span_id=parent_span_id,
            trace_id=self.trace_id,
            name=name,
            start_time=time.time(),
        )
        self.spans.append(span)
        return span

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于序列化/展示）。"""
        return {
            "trace_id": self.trace_id,
            "metadata": self.metadata,
            "total_duration_ms": round((self.end_time or time.time()) - self.start_time, 2) * 1000,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "spans": [s.to_dict() for s in self.spans],
            "span_count": len(self.spans),
            # 构建调用树摘要
            "timeline": self._build_timeline(),
        }

    def _build_timeline(self) -> List[Dict[str, Any]]:
        """构建按时间排序的事件时间线。"""
        events = []
        for span in self.spans:
            events.append({
                "time_offset_ms": round((span.start_time - self.start_time) * 1000, 2),
                "name": span.name,
                "status": span.status,
                "duration_ms": round(span.duration_ms, 2),
            })
        events.sort(key=lambda e: e["time_offset_ms"])
        return events


class Tracer:
    """全局 Trace 管理器。

    提供便捷的 span 创建方法，自动从当前上下文获取 trace_id。

    使用方式：
        # 在已有 TraceContext 中
        with tracer.span("tool_call", tags={"tool": "list_shops_by_type"}) as span:
            result = do_something()
            span.set_tag("result_count", result["count"])
    """

    def span(self, name: str, tags: Dict[str, Any] = None, parent: Span = None) -> "_SpanContextManager":
        """创建 span 上下文管理器。

        Args:
            name: span 名称
            tags: 初始标签
            parent: 父 span（可选）

        Returns:
            SpanContextManager 上下文管理器
        """
        return _SpanContextManager(name=name, tags=tags, parent=parent)

    @staticmethod
    def get_current_trace() -> Optional[TraceContext]:
        """获取当前线程的 trace 上下文。"""
        return _current_trace.get()

    @staticmethod
    def get_current_trace_id() -> Optional[str]:
        """获取当前 trace_id。"""
        ctx = _current_trace.get()
        return ctx.trace_id if ctx else None


class _SpanContextManager:
    """Span 上下文管理器：自动开始和结束 span。"""

    def __init__(self, name: str, tags: Dict[str, Any] = None, parent: Span = None):
        self.name = name
        self.tags = tags or {}
        self.parent = parent
        self._span: Optional[Span] = None
        self._trace_ctx: Optional[TraceContext] = None

    def __enter__(self) -> Span:
        self._trace_ctx = _current_trace.get()
        if self._trace_ctx:
            parent_id = self.parent.span_id if self.parent else None
            self._span = self._trace_ctx.create_span(self.name, parent_span_id=parent_id)
            for k, v in self.tags.items():
                self._span.set_tag(k, v)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            if exc_type:
                self._span.mark_error(str(exc_val))
            else:
                self._span.mark_success()


# 全局单例
tracer = Tracer()


# ==================== 便捷工具函数 ====================

def get_trace_id() -> str:
    """获取当前 trace_id，如果没有则返回 'no-trace'。"""
    return Tracer.get_current_trace_id() or "no-trace"


def log_with_trace(level: int, msg: str, **kwargs) -> None:
    """带 trace_id 的结构化日志。

    Args:
        level: 日志级别
        msg: 日志消息
        **kwargs: 额外的键值对
    """
    trace_id = get_trace_id()
    extra = {"trace_id": trace_id, **kwargs}
    logger.log(level, f"[{trace_id}] {msg}", extra=extra)


# ==================== 第五层：指标聚合 ====================

class MetricsCollector:
    """指标聚合器（第五层可观测）。

    把 trace 里分散的 span 汇聚成可监控的核心指标：

    | 指标 | 真实数据来源 |
    |------|-------------|
    | 工具延迟 P95 | 每次工具执行的实际耗时（parallel_tools 埋点） |
    | 失败率 | 执行失败/超时次数 ÷ 工具调用总次数 |
    | 重试次数 | 读操作自动重试的实际发生次数 |
    | 误触发率 | 返回空/无效结果的调用次数 ÷ 工具调用总次数（ResultQualityChecker 判定） |
    | 漏调用率 | 审计器判定"应调未调"的查询数 ÷ 参与审计的查询总数 |
    | 单任务 token 成本 / 总耗时 | trace 汇总（TraceContext.to_dict） |

    所有指标均由真实运行时数据计算，无任何模拟值。
    """

    MAX_SAMPLES = 2000

    def __init__(self):
        self._latencies: deque = deque(maxlen=self.MAX_SAMPLES)
        self._tool_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "success": 0, "failed": 0, "retries": 0, "empty": 0}
        )
        self._total_calls = 0
        self._total_retries = 0
        self._total_failed = 0
        self._total_empty = 0
        self._audit_queries = 0
        self._miss_detected = 0
        self._write_calls = 0
        self._write_unknown = 0

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        retries: int = 0,
        is_write: bool = False,
        status: str = "success",
        empty_result: bool = False,
    ) -> None:
        """记录一次工具调用（由 ParallelToolNode 埋点调用）。"""
        if not settings.metrics_enabled:
            return
        self._total_calls += 1
        self._latencies.append(float(duration_ms))
        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            self._total_failed += 1
        if retries:
            stats["retries"] += retries
            self._total_retries += retries
        if empty_result:
            stats["empty"] += 1
            self._total_empty += 1
        if is_write:
            self._write_calls += 1
            if status == "unknown":
                self._write_unknown += 1

    def record_audit(
        self,
        has_miss: bool,
        has_extra: bool = False,
    ) -> None:
        """记录一次审计结果（由 ToolCallAuditor 埋点调用）。

        Args:
            has_miss: 是否检出"应调未调"（漏调用）
            has_extra: 是否检出"不该调却调了"（误触发）
        """
        if not settings.metrics_enabled:
            return
        self._audit_queries += 1
        if has_miss:
            self._miss_detected += 1

    def record_empty_result(self, tool_name: str) -> None:
        """记录一次空/无效结果（误触发信号）。

        单独计数，不与 record_tool_call 的 calls 重复累加。
        """
        if not settings.metrics_enabled:
            return
        self._tool_stats[tool_name]["empty"] += 1
        self._total_empty += 1

    @staticmethod
    def _percentile(samples: List[float], p: float) -> float:
        """计算分位数（线性插值）。"""
        if not samples:
            return 0.0
        ordered = sorted(samples)
        idx = (len(ordered) - 1) * p
        low = int(idx)
        high = min(low + 1, len(ordered) - 1)
        weight = idx - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    def get_metrics(self) -> Dict[str, Any]:
        """输出核心监控指标。"""
        latencies = list(self._latencies)
        total = self._total_calls

        return {
            # 工具延迟
            "latency_p50_ms": round(self._percentile(latencies, 0.50), 2),
            "latency_p95_ms": round(self._percentile(latencies, 0.95), 2),
            "latency_p99_ms": round(self._percentile(latencies, 0.99), 2),
            "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
            # 失败率 / 重试次数
            "tool_calls_total": total,
            "tool_failure_rate": f"{(self._total_failed / total * 100):.2f}%" if total else "0%",
            "tool_retry_total": self._total_retries,
            "tool_retry_rate": f"{(self._total_retries / total * 100):.2f}%" if total else "0%",
            # 调用准确率 / 误触发率 / 漏调用率
            "call_precision": f"{(1 - self._total_empty / total) * 100:.2f}%" if total else "N/A",
            "false_trigger_rate": f"{(self._total_empty / total * 100):.2f}%" if total else "N/A",
            "miss_rate": (
                f"{(self._miss_detected / self._audit_queries * 100):.2f}%"
                if self._audit_queries else "N/A"
            ),
            "audit_queries_total": self._audit_queries,
            # 写操作风险
            "write_calls_total": self._write_calls,
            "write_unknown_total": self._write_unknown,
            # 分工具明细
            "per_tool": dict(self._tool_stats),
        }

    def reset(self) -> None:
        """重置所有指标。"""
        self._latencies.clear()
        self._tool_stats.clear()
        self._total_calls = 0
        self._total_retries = 0
        self._total_failed = 0
        self._total_empty = 0
        self._audit_queries = 0
        self._miss_detected = 0
        self._write_calls = 0
        self._write_unknown = 0


_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标聚合器单例。"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector