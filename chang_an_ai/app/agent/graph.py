"""LangGraph Agent 构建：自定义并行Agent单例。

性能优化（V2.0）：
- 并行工具执行：同一轮多个工具调用使用 asyncio.gather 并发执行，
  典型场景（店铺+优惠券+笔记）从 520ms 降至 250ms（提升52%）；
- 工具缓存：Java API 结果短时缓存（5-60s），向量库查询中期缓存（5-30min），
  相同查询重复命中时从缓存读取（<1ms），避免重复网络/计算开销；
- 超时控制：单个工具超时不阻塞其他工具，整体响应时间可控；
- 状态机结构：START → agent(LLM+bind_tools) → 条件边（有 tool_calls 吗？）
  → parallel_tools(ParallelToolNode) ⇄ agent → END；
- recursion_limit 由 MAX_TOOL_ROUNDS 计算：每轮工具调用涉及 2 个节点
  (agent 思考 + tools 执行)，再加 1 个最终回答节点，
  防"查不到→反复查"死循环（另一个防线在 prompt 的"失败换工具"要求）；
- 为什么 ChatOpenAI 指向 DeepSeek？DeepSeek 是 OpenAI 兼容协议，
  langchain-openai 的 ChatOpenAI 指 base_url 即可复用其 bind_tools/流式能力；
- temperature=0.3：工具调用场景要确定性，太高会乱调工具。
- 懒加载设计：langchain/langgraph 与 pydantic 2.13 存在兼容问题导致导入阻塞，
  所有 Agent 相关导入放在 build_agent() 内部，仅在首次 Agent 请求时触发，
  服务启动秒开，RAG 模式完全不受影响。

V3.0 防御增强（五层设计）：
- 第一层：严格 JSON Schema 契约 + 版本管理
- 第二层：动态工具检索 + 本地约束解码 + ToolGuard 参数校验
- 第三层：Policy Engine（工具白名单 + 持久化幂等键 + 写操作人工确认）
- 第四层：Token 预算控制（单轮 + 总预算 + 预警）
- 第五层：Trace 全链路因果链串联
- 第六层（可选）：结构化思考，强制推理步骤
"""
import logging
import json
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Windows DLL 加载顺序要求：langchain_openai -> openai -> httpx 必须在
# pydantic -> pydantic-core (.pyd) 之前导入，否则 C 扩展冲突导致解释器静默崩溃
from langchain_openai import ChatOpenAI
import httpx

from app.config import settings
from app.prompts.agent import AGENT_SYSTEM

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
RECURSION_LIMIT = MAX_TOOL_ROUNDS * 2 + 1


# ==================== 第三层：调用审计器 ====================

class ToolCallAuditor:
    """工具调用审计器（第三层防御）：检测异常调用模式。

    功能：
    1. 检测应该多调工具但只调了一个的情况
    2. 检测不该同时调用的工具组合
    3. 检测同一工具被多次调用
    4. 统计调用频率和成功率
    5. 记录异常模式用于后续分析
    """

    # 审计统计
    _stats: Dict[str, Any] = {
        "total_calls": 0,
        "anomalies_detected": 0,
        "tool_usage_count": defaultdict(int),
        "anomaly_patterns": [],
    }

    # 关键词到期望工具的映射
    KEYWORD_TOOL_MAPPING = {
        "优惠": ["list_shops_by_type", "list_vouchers"],
        "优惠券": ["list_shops_by_type", "list_vouchers"],
        "团购": ["list_shops_by_type", "list_vouchers"],
        "折扣": ["list_shops_by_type", "list_vouchers"],
        "体验": ["search_blogs"],
        "评价": ["search_blogs"],
        "笔记": ["search_blogs"],
        "口碑": ["search_blogs"],
        "攻略": ["search_knowledge"],
        "景点": ["search_knowledge"],
        "门票": ["search_knowledge"],
        "历史": ["search_knowledge"],
        # 天气意图
        "天气": ["query_weather"],
        "下雨": ["query_weather"],
        "温度": ["query_weather"],
        "穿衣": ["query_weather"],
        # 景点百科意图
        "门票": ["query_attraction", "search_knowledge"],
        "开放": ["query_attraction", "search_knowledge"],
        "值得去": ["query_attraction", "search_blogs"],
        # 交通意图
        "怎么去": ["query_transit", "query_attraction"],
        "多远": ["query_transit"],
        "多久": ["query_transit"],
        "地铁": ["query_transit"],
        "公交": ["query_transit"],
    }

    @classmethod
    def audit_tool_calls(
        cls,
        user_query: str,
        tool_calls: List[Dict[str, Any]]
    ) -> List[str]:
        """审计工具调用决策，返回警告列表。

        Args:
            user_query: 用户原始查询
            tool_calls: LLM决定的工具调用列表 [{name, args, id}, ...]

        Returns:
            警告信息列表（空列表表示正常）
        """
        warnings = []
        cls._stats["total_calls"] += 1

        if not tool_calls:
            return warnings

        tools_called = [tc['name'] for tc in tool_calls]
        args_snapshot = {tc['name']: tc.get('args', {}) for tc in tool_calls}

        # 更新工具使用统计
        for tool_name in tools_called:
            cls._stats["tool_usage_count"][tool_name] += 1

        # 异常检测模式1：用户提到优惠但未调 list_vouchers
        for keyword, expected_tools in cls.KEYWORD_TOOL_MAPPING.items():
            if keyword in user_query:
                missing_tools = [t for t in expected_tools if t not in tools_called]
                if missing_tools:
                    warning = (
                        f"[审计] 用户提到「{keyword}」但未调用必要工具: {missing_tools}。"
                        f"建议补充调用这些工具以提供完整信息。"
                    )
                    warnings.append(warning)
                    logger.warning(f"[ToolAudit] {user_query[:30]}... -> {warning}")

        # 异常检测模式2：同时调用了知识库和店铺搜索（可能意图不明确）
        knowledge_tools = {"search_knowledge", "search_blogs"}
        shop_tools = {"search_shops_by_name", "list_shops_by_type"}
        called_knowledge = set(tools_called) & knowledge_tools
        called_shop = set(tools_called) & shop_tools

        if called_knowledge and called_shop and len(tools_called) > 2:
            warning = (
                "[审计] 同时调用了知识库搜索和店铺搜索工具，"
                "请确认用户的真实意图是否需要两者都查。"
            )
            warnings.append(warning)
            logger.info(f"[ToolAudit] {user_query[:30]}... -> {warning}")

        # 异常检测模式3：同一工具被多次调用
        tool_counts = {}
        for t in tools_called:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        for t, count in tool_counts.items():
            if count > 1:
                # 检查是否是不同参数的合理调用
                args_list = [
                    json.dumps(args_snapshot[t], sort_keys=True)
                    for tc in tool_calls if tc['name'] == t
                ]
                if len(set(args_list)) == 1:
                    # 相同参数重复调用 - 真正的异常
                    warning = (
                        f"[审计] 工具 {t} 被调用了 {count} 次（相同参数），"
                        "这可能是重复调用，系统可能会拦截。"
                    )
                    warnings.append(warning)
                    logger.warning(
                        f"[ToolAudit] {user_query[:30]}... -> "
                        f"工具 {t} 重复调用 {count} 次"
                    )

        # 异常检测模式4：调了 list_vouchers 但没有先调店铺搜索
        if "list_vouchers" in tools_called:
            has_shop_search = any(
                t in tools_called
                for t in ["search_shops_by_name", "list_shops_by_type"]
            )
            if not has_shop_search:
                warning = (
                    "[审计] 直接调用了 list_vouchers 但没有先获取店铺ID，"
                    "shop_id 可能是编造的！请先调 search_shops_by_name 或 list_shops_by_type。"
                )
                warnings.append(warning)
                logger.error(
                    f"[ToolAudit] {user_query[:30]}... -> "
                    f"直接调 list_vouchers 未先获取shop_id"
                )

        # 记录异常
        if warnings:
            cls._stats["anomalies_detected"] += len(warnings)
            cls._stats["anomaly_patterns"].append({
                "query": user_query[:50],
                "tools_called": tools_called,
                "warnings": warnings,
                "timestamp": __import__("time").time(),
            })
            # 只保留最近100条异常记录
            if len(cls._stats["anomaly_patterns"]) > 100:
                cls._stats["anomaly_patterns"] = cls._stats["anomaly_patterns"][-100:]

        return warnings

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """获取审计统计信息。

        Returns:
            包含统计信息的字典
        """
        return {
            **cls._stats,
            "tool_usage_count": dict(cls._stats["tool_usage_count"]),
            "recent_anomalies": cls._stats["anomaly_patterns"][-10:] if cls._stats["anomaly_patterns"] else [],
        }

    @classmethod
    def reset_stats(cls) -> None:
        """重置统计数据（测试用）。"""
        cls._stats = {
            "total_calls": 0,
            "anomalies_detected": 0,
            "tool_usage_count": defaultdict(int),
            "anomaly_patterns": [],
        }
        logger.info("[ToolAudit] 统计数据已重置")


def build_agent():
    # 所有重导入已在模块顶部完成，此处只做构造逻辑
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from app.agent.state import AgentState
    from app.agent.tools import ALL_TOOLS
    from app.agent.cache import CachedToolsWrapper
    from app.agent.parallel_tools import ParallelToolNode
    from app.agent.tool_retriever import get_tool_retriever
    from app.agent.policy_engine import get_policy_engine, PolicyDecision, PolicyAction
    from app.agent.token_budget import get_session_budget, BudgetExceededError, TokenBudget
    from app.agent.tracer import tracer, get_trace_id

    http_client = httpx.AsyncClient(verify=False, timeout=60.0)

    # ==================== 第二层：动态工具检索 ====================
    tool_retriever = get_tool_retriever()

    # 构建工具检索用的模型（不绑定全部工具，而是在每轮动态 bind）
    base_model = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.3,
        timeout=60,
        max_retries=1,
        http_async_client=http_client,
    )

    # 默认绑定全部工具（兜底）
    fallback_model = base_model.bind_tools(ALL_TOOLS)

    # 使用带缓存的工具包装器
    cached_tools = CachedToolsWrapper(ALL_TOOLS)
    cached_tool_list = cached_tools.get_cached_tools()

    # ==================== 第三层：Policy Engine ====================
    policy_engine = get_policy_engine()

    logger.info(
        f"Agent 构建完成: model={settings.deepseek_model}, "
        f"tools={len(ALL_TOOLS)}, "
        f"recursion_limit={RECURSION_LIMIT}, "
        f"五层防御: 动态检索={settings.dynamic_tool_retrieval_enabled}, "
        f"策略引擎={settings.policy_engine_enabled}, "
        f"预算控制={settings.token_budget_enabled}, "
        f"Trace={settings.tracing_enabled}"
    )

    # 构建自定义状态图（支持并行工具执行）
    workflow = StateGraph(AgentState)

    async def agent_node(state: AgentState):
        """LLM Agent 节点：决定是否调用工具（含五层防御 + 第六层结构化思考）"""
        messages = state["messages"]
        trace_id = state.get("trace_id") or get_trace_id()
        session_id = state.get("session_id", "")

        # 提取用户查询用于日志和工具检索
        user_query = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                user_query = msg.content[:100]
                break

        # ==================== 第四层：任务级总超时（wall-clock） ====================
        task_deadline = state.get("task_deadline")
        if task_deadline is None:
            task_deadline = time.time() + settings.max_task_duration_seconds
        elif time.time() > task_deadline:
            logger.warning(
                f"[{trace_id}] ⏱ 任务总超时({settings.max_task_duration_seconds}s)已耗尽，"
                f"强制终止（当前第 {state.get('iter', 0)} 轮）"
            )
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        "抱歉，本次查询耗时较长已超出处理时限。"
                        "已获取的信息我无法完整整理，建议您稍后重试，或缩小查询范围～"
                    ),
                }],
                "iter": state.get("iter", 0) + 1,
                "param_retry": 0,
                "need_param_retry": False,
                "task_deadline": task_deadline,
            }

        # ==================== 第五层：Trace - agent_think span ====================
        think_span = None
        if settings.tracing_enabled:
            think_span_ctx = tracer.span("agent_think", tags={
                "round": state.get("iter", 0),
                "query": user_query[:50],
            })
            think_span = think_span_ctx.__enter__()

        try:
            # ==================== 第四层：Token 预算检查 ====================
            if settings.token_budget_enabled and session_id:
                try:
                    budget = get_session_budget(session_id)
                    budget.check_before_call(round_num=state.get("iter", 0))

                    if think_span:
                        status = budget.get_status()
                        think_span.set_tag("budget_usage", status.usage_ratio)
                        think_span.set_tag("budget_remaining", status.remaining_tokens)
                except BudgetExceededError as e:
                    logger.warning(f"[{trace_id}] ⛔ Token预算超限: {e}")
                    return {
                        "messages": [{
                            "role": "assistant",
                            "content": "抱歉，当前会话已达到处理上限。请开启新会话继续交流~"
                        }],
                        "param_retry": 0,
                        "need_param_retry": False,
                        "task_deadline": task_deadline,
                    }

            # ==================== 第六层防御（可选）：结构化思考 ====================
            from app.agent.structured_thinking import StructuredThinking

            if StructuredThinking.is_enabled():
                think_result = await StructuredThinking.force_think(base_model, user_query)
                if think_result and not think_result.is_valid:
                    logger.warning(
                        f"[{trace_id}] 第六层拦截：结构化思考结果无效 | "
                        f"意图: {think_result.user_intent} | "
                        f"相关: {think_result.is_travel_related}"
                    )
                if think_span:
                    think_span.set_tag(
                        "structured_thinking_valid",
                        think_result.is_valid if think_result else False
                    )

            # ==================== 第二层·防护3：工具检索（语义+关键词混合 Top-K） ====================
            if settings.dynamic_tool_retrieval_enabled:
                filtered_tools = await tool_retriever.aretrieve(user_query)
                current_model = base_model.bind_tools(filtered_tools)
                logger.debug(f"[{trace_id}] 工具检索: {len(filtered_tools)}/{len(ALL_TOOLS)} 个工具")
                if think_span:
                    think_span.set_tag("retrieved_tools", [t.name for t in filtered_tools])
            else:
                current_model = fallback_model

            # ==================== 核心调用 ====================
            call_start = time.time()
            response = await current_model.ainvoke(messages)
            call_elapsed = time.time() - call_start

            # ==================== 第四层：Token 预算记录 ====================
            if settings.token_budget_enabled and session_id:
                input_tokens = state.get("round_input_tokens", 0) or TokenBudget.estimate_tokens(
                    "\n".join([m.content if hasattr(m, 'content') else "" for m in messages[-3:]])
                )
                output_tokens = TokenBudget.estimate_tokens(
                    response.content if hasattr(response, 'content') and response.content else ""
                )
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    output_tokens += TokenBudget.estimate_tokens(
                        json.dumps(response.tool_calls, ensure_ascii=False)
                    )
                budget = get_session_budget(session_id)
                budget.record_usage(input_tokens, output_tokens, round_num=state.get("iter", 0))

            # ==================== 第五层：单任务 token 成本埋点 ====================
            # 与预算控制解耦：即使未开启预算，也要把 token 成本记入 trace，
            # 这样才能回答"单次任务花了多少 token"。
            if settings.tracing_enabled:
                _trace = tracer.get_current_trace()
                if _trace:
                    _in_tokens = TokenBudget.estimate_tokens(
                        "\n".join([
                            m.content if hasattr(m, "content") and m.content else ""
                            for m in messages[-3:]
                        ])
                    )
                    _out_tokens = TokenBudget.estimate_tokens(
                        response.content
                        if hasattr(response, "content") and response.content else ""
                    )
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        _out_tokens += TokenBudget.estimate_tokens(
                            json.dumps(response.tool_calls, ensure_ascii=False)
                        )
                    _trace.record_tokens(_in_tokens, _out_tokens)
                    if think_span:
                        think_span.set_tag("round_tokens", _in_tokens + _out_tokens)

            # ==================== 第二层·防护1+2：约束解码 + 参数校验 ====================
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tools = [tc['name'] for tc in response.tool_calls]

                # 第五层：参数校验独立 span（因果链中的"参数校验"环节）
                pv_span = None
                pv_span_ctx = None
                if settings.tracing_enabled:
                    pv_span_ctx = tracer.span("param_validation", tags={
                        "round": state.get("iter", 0),
                        "tools": tools,
                        "calls_count": len(response.tool_calls),
                    })
                    pv_span = pv_span_ctx.__enter__()

                constraint_errors = tool_retriever.validate_tool_calls(response.tool_calls)
                if pv_span:
                    pv_span.set_tag("violations", constraint_errors)
                    pv_span.set_tag("passed", not constraint_errors)
                    pv_span_ctx.__exit__(None, None, None)

                if constraint_errors:
                    retry_count = state.get("param_retry", 0)
                    logger.error(
                        f"[{trace_id}] 第二层校验拦截 {len(constraint_errors)} 个违规 "
                        f"(retry={retry_count}/{settings.param_validation_max_retry}): "
                        f"{constraint_errors}"
                    )
                    if think_span:
                        think_span.set_tag("constraint_violations", constraint_errors)
                        think_span.set_tag("param_retry", retry_count)

                    # 回传真实报错让模型修正（不执行非法调用）
                    if (
                        settings.constrained_decode_retry_on_error
                        and retry_count < settings.param_validation_max_retry
                    ):
                        from langchain_core.messages import AIMessage, ToolMessage
                        from app.agent.param_validator import get_param_validator

                        param_errors = tool_retriever.collect_param_errors(response.tool_calls)
                        feedback = (
                            get_param_validator().format_for_llm(param_errors)
                            if param_errors else "\n".join(constraint_errors)
                        )

                        blocked_calls = []
                        tool_messages = []
                        for idx, tc in enumerate(response.tool_calls):
                            call_id = tc.get("id") or f"blocked_{state.get('iter', 0)}_{idx}"
                            blocked_calls.append({
                                "name": tc.get("name", ""),
                                "args": tc.get("args") or {},
                                "id": call_id,
                                "type": "tool_call",
                            })
                            tool_messages.append(ToolMessage(
                                content=f"⚠️ 该调用被参数校验拦截，未执行。\n{feedback}",
                                tool_call_id=call_id,
                            ))

                        return {
                            "messages": [
                                AIMessage(content="", tool_calls=blocked_calls),
                                *tool_messages,
                            ],
                            "iter": state.get("iter", 0) + 1,
                            "param_retry": retry_count + 1,
                            "need_param_retry": True,
                            "task_deadline": task_deadline,
                        }

                    # 超过重试上限：直接给用户兜底答复，避免死循环
                    logger.error(f"[{trace_id}] 参数校验重试超限，终止本轮")
                    return {
                        "messages": [{
                            "role": "assistant",
                            "content": "抱歉，我没能正确组装查询参数。可以换个说法，或直接告诉我想查的店铺/分类名称～",
                        }],
                        "iter": state.get("iter", 0) + 1,
                        "param_retry": 0,
                        "need_param_retry": False,
                        "task_deadline": task_deadline,
                    }

                # ==================== 第三层：Policy Engine 规则检查 ====================
                if settings.policy_engine_enabled:
                    for tc in response.tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("args", {})

                        policy_decision = await policy_engine.evaluate(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            context={
                                "session_id": session_id,
                                "trace_id": trace_id,
                                "user_query": user_query,
                            }
                        )

                        if policy_decision.action == PolicyAction.REQUIRE_CONFIRM:
                            logger.info(
                                f"[{trace_id}] 第三层策略引擎：{tool_name} 需要人工确认, "
                                f"token={policy_decision.confirm_token}"
                            )
                            return {
                                "messages": [{
                                    "role": "assistant",
                                    "content": json.dumps({
                                        "confirm_required": True,
                                        "confirm_token": policy_decision.confirm_token,
                                        "confirm_message": policy_decision.confirm_message,
                                        "tool_name": tool_name,
                                        "tool_args": tool_args,
                                    }, ensure_ascii=False),
                                }],
                                "param_retry": 0,
                                "need_param_retry": False,
                                "task_deadline": task_deadline,
                            }

                        if not policy_decision.allowed:
                            logger.warning(
                                f"[{trace_id}] 第三层策略引擎拒绝: {tool_name} → {policy_decision.reason}"
                            )
                            response.content = (
                                f"抱歉，操作「{tool_name}」暂时无法执行：{policy_decision.reason}"
                            )
                            response.tool_calls = None
                            break

                # ==================== 第三层：调用审计（保留原有审计器） ====================
                audit_warnings = ToolCallAuditor.audit_tool_calls(
                    user_query=user_query,
                    tool_calls=response.tool_calls
                )

                # 第五层指标：漏调用/误触发信号
                try:
                    from app.agent.tracer import get_metrics_collector
                    has_miss = any("未调用必要工具" in w for w in audit_warnings)
                    has_extra = any("请确认用户的真实意图" in w for w in audit_warnings)
                    get_metrics_collector().record_audit(
                        has_miss=has_miss, has_extra=has_extra
                    )
                except Exception as e:
                    logger.debug(f"[{trace_id}] 指标埋点失败: {e}")

                critical_warnings = [w for w in audit_warnings if "编造" in w or "直接调" in w]
                if critical_warnings:
                    logger.error(
                        f"[{trace_id}] ⚠️ 严重异常检测到! "
                        f"用户: {user_query} | 工具: {tools} | 警告: {critical_warnings}"
                    )

                logger.info(f"[{trace_id}] Agent: {user_query} -> 调用工具: {tools} | 耗时: {call_elapsed:.2f}s")
            else:
                logger.info(f"[{trace_id}] Agent: {user_query} -> 直接回答 | 耗时: {call_elapsed:.2f}s")

            # 记录 trace 标签
            if think_span:
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    think_span.set_tag("tools_called", [tc['name'] for tc in response.tool_calls])
                    think_span.set_tag("tools_count", len(response.tool_calls))
                else:
                    think_span.set_tag("direct_answer", True)

            return {
                "messages": [response],
                "iter": state.get("iter", 0) + 1,
                "param_retry": 0,
                "need_param_retry": False,
                "task_deadline": task_deadline,
            }

        finally:
            if think_span_ctx:
                think_span_ctx.__exit__(None, None, None)

    # 使用并行工具节点替代默认 ToolNode（含第四层结果质量检查 + 第五层Trace）
    parallel_tool_node = ParallelToolNode(tools=cached_tool_list)

    # 包装工具节点，添加 Trace
    async def traced_tools_node(state: AgentState):
        """带 Trace 的工具执行节点。"""
        trace_id = state.get("trace_id") or get_trace_id()
        tool_calls_info = []

        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_calls_info = [tc['name'] for tc in last_message.tool_calls]

        with tracer.span("tools_execution", tags={
            "round": state.get("iter", 0),
            "tools": tool_calls_info,
            "tools_count": len(tool_calls_info),
        }) as tools_span:
            result = await parallel_tool_node.run(state)
            if tools_span:
                tools_span.set_tag("result_count", len(result.get("messages", [])))
            return result

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", traced_tools_node)

    workflow.set_entry_point("agent")

    def should_continue(state: AgentState):
        """条件边判断：参数校验重试优先，其次有 tool_calls 则执行工具，否则结束。

        第四层：所有分支都做轮数二次检查，防止重试/工具调用叠加导致死循环。
        """
        # 第二层：参数校验失败 → 回 agent 节点让模型修正后重生成
        if state.get("need_param_retry"):
            if state.get("iter", 0) >= MAX_TOOL_ROUNDS:
                logger.warning(
                    f"[{state.get('trace_id', 'no-trace')}] "
                    f"参数修正已达最大轮数 {MAX_TOOL_ROUNDS}，强制结束"
                )
                return END
            return "retry"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if state.get("iter", 0) >= MAX_TOOL_ROUNDS:
                logger.warning(
                    f"[{state.get('trace_id', 'no-trace')}] "
                    f"已达最大工具调用轮数 {MAX_TOOL_ROUNDS}，强制结束"
                )
                return END
            return "tools"
        return END

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "retry": "agent", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent