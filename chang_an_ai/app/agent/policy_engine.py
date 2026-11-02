"""策略引擎（第三层安全治理层）：Policy Engine + 幂等键 + 人工确认。

设计要点：
1. Policy Engine：可配置的规则引擎，在工具执行前评估是否允许
2. 幂等键持久化：基于 Redis 的持久化幂等键，防重复执行（尤其写操作）
3. 人工确认：写操作必须等待用户确认后才执行（Human-in-the-loop）
4. 规则可扩展：通过策略链模式，支持动态添加/删除规则

策略链：
    请求 → Rule1(工具白名单) → Rule2(参数合规) → Rule3(频率限制) →
           Rule4(幂等检查) → Rule5(写操作确认) → 放行/拒绝

使用方式：
    from app.agent.policy_engine import PolicyEngine, PolicyDecision

    engine = PolicyEngine()

    # 执行前检查
    decision = await engine.evaluate(
        tool_name="submit_feedback",
        tool_args={"feedback_type": "complaint", "content": "test"},
        session_id="sess_123",
    )

    if not decision.allowed:
        return {"error": decision.reason}

    if decision.requires_confirmation:
        # 需要用户确认
        return {"confirm_required": True, "confirm_token": decision.confirm_token}
"""
from __future__ import annotations

import json
import time
import uuid
import hashlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """策略动作。"""
    ALLOW = "allow"           # 放行
    DENY = "deny"             # 拒绝
    REQUIRE_CONFIRM = "require_confirm"  # 需要人工确认


class ToolPermission(Enum):
    """工具权限等级（权限分级的最小模型）。

    对应"模型不能决定能不能做"：等级由后端声明，模型无权改变。
    - READ：只读，ALLOW 自动执行
    - WRITE：写操作，必须 ASK 人工确认
    - MONEY：涉及金额，必须 ASK 人工确认（比 WRITE 更严格，不允许"自动重试"）
    """
    READ = "read"
    WRITE = "write"
    MONEY = "money"


# 工具权限分级表：新增工具必须在此登记，未登记的工具会被白名单规则拒绝
TOOL_PERMISSIONS: Dict[str, ToolPermission] = {
    "search_shops_by_name": ToolPermission.READ,
    "list_shops_by_type": ToolPermission.READ,
    "get_shop_detail": ToolPermission.READ,
    "list_vouchers": ToolPermission.READ,
    "search_blogs": ToolPermission.READ,
    "search_knowledge": ToolPermission.READ,
    "submit_feedback": ToolPermission.WRITE,
}

# 涉及金额的工具（下单/支付/退款类）。当前项目未接入此类工具，
# 预留分类：一旦新增（如 create_order / pay / refund），必须登记到此集合，
# 自动获得"人工确认 + 禁止自动重试"的最严格处理。
MONEY_TOOLS: set = set()

# 需要人工确认的权限等级
CONFIRM_REQUIRED_PERMISSIONS = {ToolPermission.WRITE, ToolPermission.MONEY}


def get_tool_permission(tool_name: str) -> ToolPermission:
    """获取工具权限等级。未登记的工具按 MONEY 处理（最保守）。"""
    if tool_name in MONEY_TOOLS:
        return ToolPermission.MONEY
    return TOOL_PERMISSIONS.get(tool_name, ToolPermission.MONEY)


def is_write_tool(tool_name: str) -> bool:
    """是否写操作（含涉及金额）。写操作禁止自动重试。"""
    return get_tool_permission(tool_name) in CONFIRM_REQUIRED_PERMISSIONS


@dataclass
class PolicyDecision:
    """策略评估结果。"""
    allowed: bool
    reason: str
    action: PolicyAction = PolicyAction.ALLOW
    requires_confirmation: bool = False
    confirm_token: Optional[str] = None  # 确认令牌（用于后续确认操作）
    confirm_message: Optional[str] = None  # 展示给用户的确认提示
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        reason: str = "所有规则通过",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyDecision":
        return cls(
            allowed=True,
            reason=reason,
            action=PolicyAction.ALLOW,
            metadata=metadata or {},
        )

    @classmethod
    def deny(
        cls,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyDecision":
        return cls(
            allowed=False,
            reason=reason,
            action=PolicyAction.DENY,
            metadata=metadata or {},
        )

    @classmethod
    def require_confirm(
        cls,
        reason: str,
        confirm_message: str,
        confirm_token: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PolicyDecision":
        return cls(
            allowed=False,
            reason=reason,
            action=PolicyAction.REQUIRE_CONFIRM,
            requires_confirmation=True,
            confirm_token=confirm_token,
            confirm_message=confirm_message,
            metadata=metadata or {},
        )


class PolicyRule(ABC):
    """策略规则基类。

    所有规则继承此类并实现 evaluate 方法。
    规则通过 PolicyEngine 的策略链串联执行。
    """

    def __init__(self, name: str, priority: int = 100):
        self.name = name
        self.priority = priority  # 数值越小越先执行

    @abstractmethod
    async def evaluate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PolicyDecision:
        """评估请求是否符合规则。

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            context: 上下文信息（session_id, history, trace_id等）

        Returns:
            PolicyDecision 对象
        """
        ...


class ToolWhitelistRule(PolicyRule):
    """规则1：工具白名单校验。

    确保LLM返回的工具名在预定义的可用工具列表中。
    """

    def __init__(self):
        super().__init__(name="tool_whitelist", priority=10)
        # 可用工具白名单 = 权限分级表登记的工具
        self._valid_tools = set(TOOL_PERMISSIONS.keys())

    async def evaluate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PolicyDecision:
        if tool_name not in self._valid_tools:
            return PolicyDecision.deny(
                f"工具「{tool_name}」不在可用白名单中，可用工具: {self._valid_tools}"
            )
        return PolicyDecision.allow()


class PermissionTierRule(PolicyRule):
    """规则2：权限分级。

    只读工具直接 ALLOW 自动执行；写操作/涉及金额操作必须 ASK 人工确认。
    权限等级来自后端声明的 TOOL_PERMISSIONS，模型无法影响。
    """

    def __init__(self):
        super().__init__(name="permission_tier", priority=20)

    async def evaluate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PolicyDecision:
        permission = get_tool_permission(tool_name)
        context["_permission"] = permission.value

        if permission == ToolPermission.READ:
            return PolicyDecision.allow(
                reason=f"只读工具「{tool_name}」自动放行",
                metadata={"permission": permission.value},
            )

        if permission == ToolPermission.MONEY and not settings.policy_money_requires_confirm:
            return PolicyDecision.allow(
                reason=f"涉及金额工具「{tool_name}」未开启确认（配置关闭）",
                metadata={"permission": permission.value},
            )

        return PolicyDecision.allow(
            reason=f"「{tool_name}」权限等级 {permission.value}，进入确认流程",
            metadata={"permission": permission.value, "needs_confirm": True},
        )


class IdempotencyRule(PolicyRule):
    """规则4：幂等键检查（request_id 机制）。

    图片要求："写操作必须携带 request_id，重试时下游能去重，避免重复扣款/下单"。

    因此这里做两件事，缺一不可：
    1. **生成并注入 request_id 到 tool_args** —— 只在上游记录幂等键是不够的，
       下游（Java 业务服务）拿不到 request_id 就无法去重。
    2. **用 request_id 作为幂等键**，而非哈希业务参数。

    request_id 必须是"稳定可复现"的：同一会话内、同一工具、同一业务参数
    重试时得到相同的 request_id，下游才能识别为同一笔操作。

    读操作 vs 写操作的重复处理不同：
    - 读操作重复 → 直接 DENY（省资源，重试无意义）
    - 写操作重复 → 放行但标记 is_retry，交给下游按 request_id 去重
      （若在网关直接 DENY，下游超时但已执行时重试会永远失败，反而更危险）
    """

    READ_IDEMPOTENCY_TTL = 5       # 读操作幂等键有效期（秒）
    WRITE_IDEMPOTENCY_TTL = 300    # 写操作幂等键有效期（秒）

    def __init__(self):
        super().__init__(name="idempotency", priority=40)

    @staticmethod
    def make_request_id(tool_name: str, tool_args: Dict[str, Any], session_id: str) -> str:
        """生成稳定可复现的 request_id。

        排除 request_id 自身（避免自引用），保证同一业务操作重试时 ID 不变。

        Args:
            tool_name: 工具名
            tool_args: 工具参数
            session_id: 会话 ID

        Returns:
            形如 req_ab12cd34ef56 的幂等请求 ID
        """
        payload = {k: v for k, v in (tool_args or {}).items() if k != "request_id"}
        key_data = f"{session_id}:{tool_name}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
        return f"req_{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"

    def _make_idempotency_key(self, request_id: str) -> str:
        """幂等键直接以 request_id 为准（下游同一 ID 即同一笔操作）。"""
        return f"idem:req:{request_id}"

    def _check_redis(self, key: str, ttl: int) -> bool:
        """在 Redis 中检查幂等键是否存在。

        Returns:
            True 表示重复（已存在），False 表示首次调用
        """
        if not settings.idempotency_persist:
            # 不持久化时回退到 ToolGuard 的内存检测
            return False
        try:
            import redis
            r = redis.Redis.from_url(
                settings.redis_url,
                password=settings.redis_password or None,
                socket_connect_timeout=1,
            )
            # SET NX：仅当 key 不存在时设置
            result = r.set(key, str(int(time.time())), nx=True, ex=ttl)
            return result is None  # None 表示 key 已存在
        except Exception as e:
            logger.warning(f"[Idempotency] Redis 检查失败: {e}")
            return False

    async def evaluate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PolicyDecision:
        is_write = is_write_tool(tool_name)
        ttl = self.WRITE_IDEMPOTENCY_TTL if is_write else self.READ_IDEMPOTENCY_TTL

        # 若模型/前端已显式给出 request_id，尊重它（支持跨请求重放）
        request_id = tool_args.get("request_id") or self.make_request_id(
            tool_name, tool_args, context.get("session_id", "")
        )
        key = self._make_idempotency_key(request_id)

        is_duplicate = self._check_redis(key, ttl)

        # 关键：把 request_id 注入参数，下游才能按它去重。
        # 仅对写操作注入——读工具的 Schema 未声明该参数，注入会导致参数校验失败。
        if is_write and settings.idempotency_inject_request_id:
            tool_args["request_id"] = request_id

        context["_request_id"] = request_id
        context["_idempotency_key"] = key

        if is_duplicate and not is_write:
            category = "读操作"
            return PolicyDecision.deny(
                f"⏰ {category}「{tool_name}」已在短时间内执行过相同参数的调用，"
                f"幂等键: {key}，拒绝重复执行"
            )

        if is_duplicate and is_write:
            logger.warning(
                f"[Idempotency] 写操作重试: {tool_name}, request_id={request_id}，"
                f"已放行，交由下游按 request_id 去重"
            )
            return PolicyDecision.allow(
                reason=f"写操作重试，下游将按 request_id={request_id} 去重",
                metadata={
                    "request_id": request_id,
                    "idempotency_key": key,
                    "is_retry": True,
                },
            )

        return PolicyDecision.allow(
            metadata={"request_id": request_id, "idempotency_key": key, "is_retry": False}
        )


class WriteConfirmRule(PolicyRule):
    """规则5：写操作人工确认。

    对 submit_feedback 等写操作，要求先返回确认提示给用户，
    用户确认后才真正执行。实现 Human-in-the-loop 模式。

    确认流程：
    1. LLM 决定调 submit_feedback → Policy Engine 拦截
    2. 返回 confirm_required + confirm_token 给前端
    3. 用户看到确认提示，点击确认
    4. 前端带 confirm_token 重新请求 → 实际执行
    """

    # 待确认操作暂存（内存，生产环境建议 Redis + TTL）
    _pending_confirms: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        super().__init__(name="write_confirm", priority=50)

    async def evaluate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PolicyDecision:
        permission = get_tool_permission(tool_name)
        if permission not in CONFIRM_REQUIRED_PERMISSIONS:
            return PolicyDecision.allow()

        # 涉及金额的操作必须确认（更严格的开关）
        if permission == ToolPermission.MONEY:
            if not settings.policy_money_requires_confirm:
                return PolicyDecision.allow()
        elif not settings.policy_write_requires_confirm:
            return PolicyDecision.allow()

        # 检查是否是确认后的重放（带 confirm_token）
        confirm_token = tool_args.pop("_confirm_token", None)
        if confirm_token and confirm_token in self._pending_confirms:
            pending = self._pending_confirms.pop(confirm_token)
            logger.info(f"[WriteConfirm] 确认令牌验证通过: {confirm_token}")
            return PolicyDecision.allow(
                reason=f"用户已确认执行 {tool_name}",
                metadata={"confirmed": True, "original_args": pending.get("args")}
            )

        # 需要确认：生成确认令牌并暂存
        token = f"confirm_{uuid.uuid4().hex[:12]}"
        msg = self._build_confirm_message(tool_name, tool_args)

        self._pending_confirms[token] = {
            "tool_name": tool_name,
            "args": tool_args.copy(),
            "session_id": context.get("session_id", ""),
            "created_at": time.time(),
        }

        # 清理过期确认（300秒超时）
        self._cleanup_expired()

        logger.info(
            f"[WriteConfirm] 写操作需要人工确认: {tool_name}, "
            f"token={token}, session={context.get('session_id', 'unknown')[:8]}..."
        )

        return PolicyDecision.require_confirm(
            reason=f"写操作「{tool_name}」需要用户确认后执行",
            confirm_message=msg,
            confirm_token=token,
        )

    def _build_confirm_message(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """构建展示给用户的确认消息。"""
        if tool_name == "submit_feedback":
            fb_type = tool_args.get("feedback_type", "其他")
            type_names = {
                "complaint": "投诉",
                "voucher_issue": "优惠券问题",
                "suggestion": "建议",
                "bug": "故障报告",
                "other": "其他",
            }
            type_cn = type_names.get(fb_type, fb_type)
            content = tool_args.get("content", "")[:50]
            shop = tool_args.get("shop_name", "")
            shop_part = f"（店铺：{shop}）" if shop else ""
            return (
                f"确认提交以下反馈？\n"
                f"类型：{type_cn}\n"
                f"内容：{content}...\n"
                f"{shop_part}\n\n"
                f"点击确认后提交。"
            )
        return f"确认执行「{tool_name}」操作？"

    def _cleanup_expired(self):
        """清理过期的待确认记录。"""
        now = time.time()
        expired = [
            token for token, data in self._pending_confirms.items()
            if now - data["created_at"] > 300
        ]
        for token in expired:
            del self._pending_confirms[token]

    @classmethod
    def get_pending_count(cls) -> int:
        """获取待确认操作数量（监控用）。"""
        return len(cls._pending_confirms)


class PolicyEngine:
    """策略引擎：串联执行多条规则，任一拒绝则整体拒绝。

    采用策略链模式：按优先级排序后依次执行规则。
    - 任一规则返回 DENY → 立即终止，返回拒绝
    - 任一规则返回 REQUIRE_CONFIRM → 返回确认要求
    - 全部 ALLOW → 放行

    使用方式：
        engine = PolicyEngine()
        engine.add_rule(ToolWhitelistRule())
        engine.add_rule(IdempotencyRule())

        decision = await engine.evaluate(tool_name, tool_args, context)
    """

    def __init__(self):
        self._rules: List[PolicyRule] = []
        self._stats = {
            "total_checks": 0,
            "allowed": 0,
            "denied": 0,
            "required_confirm": 0,
        }
        # 审计证据链：谁调用、调什么、参数、决策依据、执行结果
        self._audit_trail: deque = deque(maxlen=1000)

    def _record_audit(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        ctx: Dict[str, Any],
        action: PolicyAction,
        reason: str,
        rule_name: str,
    ) -> None:
        """写入一条审计记录（执行结果稍后由 record_execution_result 回填）。"""
        try:
            self._audit_trail.append({
                "ts": time.time(),
                "trace_id": ctx.get("trace_id"),
                "session_id": ctx.get("session_id"),
                # 谁调用
                "actor": ctx.get("user_id") or ctx.get("session_id"),
                # 调什么
                "tool": tool_name,
                # 参数
                "args": dict(tool_args or {}),
                # 决策依据
                "permission": ctx.get("_permission"),
                "request_id": ctx.get("_request_id"),
                "decision": action.value,
                "rule": rule_name,
                "reason": reason,
                # 执行结果（回填）
                "status": "pending",
                "result": None,
                "duration_ms": None,
            })
        except Exception as e:
            logger.warning(f"[PolicyAudit] 写入审计记录失败: {e}")

    def record_execution_result(
        self,
        tool_name: str,
        request_id: Optional[str] = None,
        status: str = "success",
        result: Any = None,
        duration_ms: Optional[float] = None,
    ) -> bool:
        """回填执行结果，闭合审计证据链。

        Args:
            tool_name: 工具名
            request_id: 幂等请求 ID（优先按它精确匹配）
            status: success | failed | timeout | unknown | denied
            result: 执行结果（会被截断存储）
            duration_ms: 执行耗时（毫秒）

        Returns:
            是否找到并回填了审计记录
        """
        for record in reversed(self._audit_trail):
            if record.get("status") != "pending":
                continue
            if record.get("tool") != tool_name:
                continue
            if request_id and record.get("request_id") != request_id:
                continue
            record["status"] = status
            record["duration_ms"] = duration_ms
            text = str(result) if result is not None else ""
            record["result"] = text[:500]
            return True
        return False

    def get_audit_trail(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """查询审计证据链。"""
        records = list(self._audit_trail)
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        return records[-limit:]

    def get_audit_summary(self) -> Dict[str, Any]:
        """审计统计摘要（谁、调了什么、决策分布）。"""
        records = list(self._audit_trail)
        tool_counts: Dict[str, int] = defaultdict(int)
        decision_counts: Dict[str, int] = defaultdict(int)
        status_counts: Dict[str, int] = defaultdict(int)
        for r in records:
            tool_counts[r.get("tool", "unknown")] += 1
            decision_counts[r.get("decision", "unknown")] += 1
            status_counts[r.get("status", "unknown")] += 1
        return {
            "total_records": len(records),
            "tool_counts": dict(tool_counts),
            "decision_counts": dict(decision_counts),
            "status_counts": dict(status_counts),
        }

    def add_rule(self, rule: PolicyRule) -> None:
        """添加规则并保持按优先级排序。"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
        logger.info(f"[PolicyEngine] 已添加规则: {rule.name} (priority={rule.priority})")

    def remove_rule(self, rule_name: str) -> bool:
        """移除指定名称的规则。"""
        for r in self._rules:
            if r.name == rule_name:
                self._rules.remove(r)
                logger.info(f"[PolicyEngine] 已移除规则: {rule_name}")
                return True
        return False

    async def evaluate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> PolicyDecision:
        """评估工具调用请求。

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            context: 上下文（session_id, trace_id, history 等）

        Returns:
            PolicyDecision 对象
        """
        ctx = context or {}
        self._stats["total_checks"] += 1

        logger.debug(f"[PolicyEngine] 评估: {tool_name}, rules={len(self._rules)}")

        for rule in self._rules:
            try:
                decision = await rule.evaluate(tool_name, tool_args, ctx)

                if not decision.allowed and decision.action == PolicyAction.DENY:
                    self._stats["denied"] += 1
                    self._record_audit(
                        tool_name, tool_args, ctx,
                        PolicyAction.DENY, decision.reason, rule.name,
                    )
                    logger.warning(
                        f"[PolicyEngine] 被「{rule.name}」拒绝: {decision.reason}"
                    )
                    return decision

                if decision.action == PolicyAction.REQUIRE_CONFIRM:
                    self._stats["required_confirm"] += 1
                    self._record_audit(
                        tool_name, tool_args, ctx,
                        PolicyAction.REQUIRE_CONFIRM, decision.reason, rule.name,
                    )
                    logger.info(
                        f"[PolicyEngine] 「{rule.name}」要求确认: {tool_name}"
                    )
                    return decision

            except Exception as e:
                logger.error(f"[PolicyEngine] 规则「{rule.name}」执行异常: {e}，跳过此规则")
                continue

        self._stats["allowed"] += 1
        self._record_audit(
            tool_name, tool_args, ctx,
            PolicyAction.ALLOW, "所有规则通过", "policy_chain",
        )
        logger.debug(f"[PolicyEngine] {tool_name} 所有规则通过，放行")
        return PolicyDecision.allow()

    def get_stats(self) -> Dict[str, Any]:
        """获取策略引擎统计信息。"""
        total = self._stats["total_checks"]
        return {
            **self._stats,
            "rules_count": len(self._rules),
            "deny_rate": f"{(self._stats['denied'] / total * 100):.1f}%" if total > 0 else "0%",
        }

    @staticmethod
    def build_default() -> "PolicyEngine":
        """构建默认的策略引擎（包含所有标准规则）。"""
        engine = PolicyEngine()
        engine.add_rule(ToolWhitelistRule())      # 白名单（priority=10）
        engine.add_rule(PermissionTierRule())     # 权限分级（priority=20）
        engine.add_rule(IdempotencyRule())        # 幂等键（priority=40）
        engine.add_rule(WriteConfirmRule())       # 写操作确认（priority=50）
        return engine


# 全局 PolicyEngine 实例（懒加载）
_policy_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """获取全局 PolicyEngine 实例。"""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine.build_default()
    return _policy_engine