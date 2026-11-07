"""Phase 2: 多 Agent 协调器

将单体 Agent 拆分为 4 个专业化子 Agent + 1 个协调器。

架构:
  ShopAgent       ─ search_shops_by_name / list_shops_by_type / get_shop_detail
  VoucherAgent    ─ list_vouchers
  BlogAgent       ─ search_blogs
  KnowledgeAgent  ─ search_knowledge
  Coordinator     ─ 意图分发 + 并行调度 + 结果合并

调用流程:
  1. chat() 路由到 agent → route_result
  2. MultiAgentCoordinator.run(messages, route_result)
  3. 单意图: 选一个子 Agent 流式执行
  4. 多意图: 主 Agent 流式 + 辅 Agent 后台执行 → 结果合并追加
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """单个子 Agent 的运行任务"""
    name: str
    intent: str
    system_prompt: str
    tools: List[Any]
    query: str
    history: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AgentResult:
    """子 Agent 的执行结果"""
    agent_name: str
    intent: str
    success: bool
    answer: str = ""
    tool_calls_count: int = 0
    elapsed_ms: float = 0
    error: str = ""


class MultiAgentCoordinator:
    """多 Agent 协调器。

    职责:
    1. 根据 route_result 的意图分发到子 Agent
    2. 单意图时用流式 Agent 运行（复用六层防御）
    3. 多意图时主 Agent 流式 + 辅 Agent 并行后台运行
    4. 合并多 Agent 结果为统一回答
    """

    # 路由值 → 子 Agent 映射（来自 routing_config_v2 的关键词归属）
    INTENT_TO_AGENT = {
        "agent": ("shop", "店铺搜索Agent"),
        "shop": ("shop", "店铺搜索Agent"),
        "voucher": ("voucher", "优惠券Agent"),
        "blog": ("blog", "笔记评价Agent"),
        "rag": ("knowledge", "知识攻略Agent"),
        "knowledge": ("knowledge", "知识攻略Agent"),
        "service": ("service", "客服Agent"),
        "chat": ("chat", "闲聊Agent"),
        "reject": ("reject", "拒答Agent"),
    }

    # 次要意图到 agent 类型的映射（用于多意图）
    SECONDARY_INTENT_FALLBACK = {
        "shop": "shop",
        "voucher": "voucher",
        "blog": "blog",
        "knowledge": "knowledge",
        "agent": "shop",  # agent 太通用，fallback 到 shop
        "rag": "knowledge",
    }

    def __init__(self):
        self._model = None

    @property
    def model(self):
        """懒加载 ChatOpenAI 实例（与 graph.py 共享模型配置）"""
        if self._model is None:
            from langchain_openai import ChatOpenAI
            from app.config import settings
            self._model = ChatOpenAI(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=0.3,
                timeout=60,
                max_retries=1,
            )
        return self._model

    # ==================== 主入口 ====================

    async def run(
        self,
        messages: List[tuple],
        route_result: Any,
        stream: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """主入口：根据路由结果分发到子 Agent。

        Args:
            messages: (role, content) 元组列表
            route_result: RouteResult 对象（含 route, is_multi_intent, secondary_intents）
            stream: 是否流式输出（StreamingResponse 场景为 True）

        Yields:
            SSE 事件字典 {"event": "...", "data": ...}
        """
        from app.agent.tools import AGENT_TOOL_MAP, ALL_TOOLS
        from app.prompts.agent import AGENT_PROMPT_MAP, COORDINATOR_MERGE_PROMPT
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

        route_val = route_result.route.value
        is_multi = route_result.is_multi_intent
        secondary = route_result.secondary_intents if is_multi else []

        user_query = messages[-1][1] if messages else ""

        logger.info(
            f"[Coordinator] route={route_val} multi={is_multi} "
            f"secondary={[s.get('route','') for s in secondary]} "
            f"msg='{user_query[:30]}...'"
        )

        if not is_multi:
            # 单意图：直接选一个子 Agent 流式运行
            async for event in self._run_primary_agent(messages, route_val, user_query):
                yield event
        else:
            # 多意图：主 Agent 流式 + 辅 Agent 后台并行
            async for event in self._run_multi_agent(messages, route_val, secondary, user_query):
                yield event

    # ==================== 单意图 Agent ====================

    async def _run_primary_agent(
        self,
        messages: List[tuple],
        route_val: str,
        user_query: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """运行单意图 Agent（流式）。"""
        agent_info = self.INTENT_TO_AGENT.get(route_val)
        if not agent_info:
            logger.warning(f"[Coordinator] 未知路由: {route_val}, fallback to shop")
            agent_name, intent = "shop", "shop"
        else:
            intent, agent_name = agent_info

        system_prompt = AGENT_PROMPT_MAP.get(intent, AGENT_PROMPT_MAP["shop"])
        tools = AGENT_TOOL_MAP.get(intent, AGENT_TOOL_MAP["shop"])

        logger.info(f"[Coordinator] 单意图 Agent: {agent_name} | tools={[t.name for t in tools]}")

        bound_model = self.model.bind_tools(tools)
        conversation = [SystemMessage(content=system_prompt)]

        for role, content in messages:
            if role == "user" or role == "human":
                conversation.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                conversation.append(AIMessage(content=content))
            else:
                conversation.append(HumanMessage(content=content))

        max_rounds = 6
        tool_call_total = 0

        for round_num in range(max_rounds):
            try:
                response = await bound_model.ainvoke(conversation)
            except Exception as e:
                logger.error(f"[Coordinator] LLM 调用失败: {e}")
                yield {"event": "error", "data": {"message": f"Agent 调用出错: {e}"}}
                return

            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls_info = [
                    {"name": tc["name"], "input": tc.get("args", {})}
                    for tc in response.tool_calls
                ]
                for tc in tool_calls_info:
                    yield {"event": "tool_call", "data": tc}

                conversation.append(response)

                tool_results = []
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})

                    tool_fn = None
                    for t in tools:
                        if t.name == tool_name:
                            tool_fn = t
                            break
                    if tool_fn is None:
                        for t in ALL_TOOLS:
                            if t.name == tool_name:
                                tool_fn = t
                                break

                    if tool_fn:
                        try:
                            result = await tool_fn.ainvoke(tool_args)
                            if isinstance(result, str):
                                result_content = result
                            else:
                                result_content = json.dumps(result, ensure_ascii=False, default=str)
                        except Exception as e:
                            result_content = json.dumps({"error": str(e)})
                        tool_call_total += 1
                    else:
                        result_content = json.dumps({"error": f"未找到工具: {tool_name}"})

                    tool_results.append(ToolMessage(
                        content=result_content,
                        tool_call_id=tc["id"],
                    ))

                conversation.extend(tool_results)
                continue

            answer = response.content if hasattr(response, "content") else ""
            if answer:
                yield {"event": "delta", "data": answer}
            yield {
                "event": "done",
                "data": {"agent": agent_name, "tools_called": tool_call_total}
            }
            return

        yield {"event": "delta", "data": "抱歉，暂时没能查到结果，可以换个说法再试试～"}
        yield {"event": "done", "data": {"agent": agent_name, "warning": "max_rounds"}}

    # ==================== 多意图 Agent ====================

    async def _run_multi_agent(
        self,
        messages: List[tuple],
        primary_route: str,
        secondary_intents: List[Dict],
        user_query: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """多意图处理：主 Agent 流式 + 辅 Agent 后台并行 → 合并追加。

        策略:
        1. 主 Agent 按正常流式输出（用户先看到主要答案）
        2. 辅 Agent 在后台并行运行（不阻塞主流程）
        3. 主 Agent 结束后，将辅 Agent 结果以 multi_answer 事件追加
        """
        # 解析次要意图
        secondary_tasks: List[AgentTask] = []
        for sec in secondary_intents[:2]:
            sec_route = sec.get("route", "")
            sec_segment = sec.get("segment", "")
            if not sec_segment:
                continue

            agent_intent = self.SECONDARY_INTENT_FALLBACK.get(sec_route, "knowledge")
            if agent_intent == self._primary_to_intent(primary_route):
                continue

            task = AgentTask(
                name=f"secondary_{agent_intent}",
                intent=agent_intent,
                system_prompt=AGENT_PROMPT_MAP.get(agent_intent, AGENT_PROMPT_MAP["knowledge"]),
                tools=AGENT_TOOL_MAP.get(agent_intent, AGENT_TOOL_MAP["knowledge"]),
                query=sec_segment,
            )
            secondary_tasks.append(task)

        if not secondary_tasks:
            async for event in self._run_primary_agent(messages, primary_route, user_query):
                yield event
            return

        logger.info(
            f"[Coordinator] 多意图: primary={primary_route} "
            f"secondary={[(t.intent, t.query[:20]) for t in secondary_tasks]}"
        )

        # 启动辅 Agent 后台任务
        secondary_future = asyncio.ensure_future(
            self._run_secondary_agents(secondary_tasks)
        )

        # 主 Agent 流式输出
        collected_primary = []
        has_done = False
        async for event in self._run_primary_agent(messages, primary_route, user_query):
            if event["event"] == "delta":
                collected_primary.append(event["data"])
            elif event["event"] == "done":
                has_done = True
                yield event
            else:
                yield event

        # 等待辅 Agent 完成（以主 Agent done 为超时起点，最多再等 10s）
        try:
            secondary_results = await asyncio.wait_for(secondary_future, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("[Coordinator] 辅 Agent 超时，跳过次要结果合并")
            secondary_results = []
        except Exception as e:
            logger.error(f"[Coordinator] 辅 Agent 异常: {e}")
            secondary_results = []

        # 合并结果
        if secondary_results:
            primary_text = "".join(collected_primary)
            merged_text = await self._merge_results(primary_text, secondary_results)
            if merged_text and merged_text != primary_text:
                yield {
                    "event": "multi_answer",
                    "data": {
                        "text": merged_text,
                        "secondary_count": len(secondary_results),
                        "secondary_intents": [r.intent for r in secondary_results],
                    }
                }

    async def _run_secondary_agents(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """并行运行多个辅 Agent（非流式）。"""
        if not tasks:
            return []

        async def run_single(task: AgentTask) -> AgentResult:
            start = time.perf_counter()
            try:
                bound = self.model.bind_tools(task.tools)
                conv = [SystemMessage(content=task.system_prompt)]
                conv.append(HumanMessage(content=task.query))

                tool_calls_count = 0
                for _ in range(4):
                    response = await bound.ainvoke(conv)

                    if hasattr(response, "tool_calls") and response.tool_calls:
                        conv.append(response)
                        for tc in response.tool_calls:
                            tool_fn = next(
                                (t for t in task.tools if t.name == tc["name"]), None
                            )
                            if tool_fn is None:
                                tool_fn = next(
                                    (t for t in ALL_TOOLS if t.name == tc["name"]), None
                                )
                            if tool_fn:
                                try:
                                    result = await tool_fn.ainvoke(tc.get("args", {}))
                                    result_content = (
                                        result if isinstance(result, str)
                                        else json.dumps(result, ensure_ascii=False, default=str)
                                    )
                                except Exception as e:
                                    result_content = json.dumps({"error": str(e)})
                            else:
                                result_content = json.dumps({"error": f"unknown tool: {tc['name']}"})
                            conv.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))
                            tool_calls_count += 1
                        continue

                    elapsed = (time.perf_counter() - start) * 1000
                    answer = response.content if hasattr(response, "content") else ""
                    return AgentResult(
                        agent_name=task.name,
                        intent=task.intent,
                        success=True,
                        answer=answer,
                        tool_calls_count=tool_calls_count,
                        elapsed_ms=elapsed,
                    )

                answer = response.content if hasattr(response, "content") else ""
                elapsed = (time.perf_counter() - start) * 1000
                return AgentResult(
                    agent_name=task.name,
                    intent=task.intent,
                    success=True,
                    answer=answer,
                    tool_calls_count=tool_calls_count,
                    elapsed_ms=elapsed,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                return AgentResult(
                    agent_name=task.name,
                    intent=task.intent,
                    success=False,
                    error=str(e),
                    elapsed_ms=elapsed,
                )

        results = await asyncio.gather(*(run_single(t) for t in tasks))
        for r in results:
            logger.info(
                f"[Coordinator] 辅Agent {r.agent_name}: "
                f"success={r.success} tools={r.tool_calls_count} "
                f"elapsed={r.elapsed_ms:.0f}ms"
            )
        return [r for r in results if r.success and r.answer]

    async def _merge_results(
        self,
        primary_text: str,
        secondary_results: List[AgentResult],
    ) -> str:
        """使用协调器 Prompt 合并主/辅 Agent 结果为统一回答。

        如果合并失败（LLM 异常），直接拼接原始结果作为兜底。
        """
        if not secondary_results:
            return primary_text

        secondary_texts = "\n".join([
            f"[{r.intent}] {r.answer}" for r in secondary_results
        ])

        merge_input = COORDINATOR_MERGE_PROMPT.format(
            primary_result=primary_text,
            secondary_results=secondary_texts,
        )

        try:
            merge_model = self.model  # 不 bind tools，纯文本合并
            conv = [
                HumanMessage(content=f"""用户的原问题涉及多个方面，需要你整合答案。

主要问题的回答：
{primary_text}

次要问题的回答：
{secondary_texts}

请基于以上信息，整合成一个连贯的回答。先回答主要问题，再补充次要信息。
注意：不要编造任何信息，只引用上面已有的内容。""")
            ]
            response = await merge_model.ainvoke(conv)
            merged = response.content if hasattr(response, "content") else ""
            if merged and len(merged) > 20:
                return merged
        except Exception as e:
            logger.error(f"[Coordinator] 合并失败: {e}")

        # 兜底：简单拼接
        parts = [primary_text]
        for r in secondary_results:
            parts.append(f"\n\n另外，{r.answer}")
        return "".join(parts)

    # ==================== 辅助方法 ====================

    def _primary_to_intent(self, route_val: str) -> str:
        """将路由值映射为 agent intent 键。"""
        mapping = {
            "agent": "shop",
            "rag": "knowledge",
            "service": "service",
        }
        return mapping.get(route_val, route_val)


# ==================== 单例 ====================

_coordinator: Optional[MultiAgentCoordinator] = None


def get_coordinator() -> MultiAgentCoordinator:
    """获取多 Agent 协调器单例。"""
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiAgentCoordinator()
    return _coordinator