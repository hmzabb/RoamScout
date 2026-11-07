"""动态工具检索器（第二层·防护三）：语义 + 关键词混合检索，筛出 Top-K 候选池。

概念澄清——工具检索解决的是"工具规模爆炸"问题：
工具只有个位数时，全量塞给模型即可；但工具到几十上百个后，全量塞入会
稀释模型注意力、抬高 token 成本、降低选择准确率。业界做法是先用检索
把候选池收敛到 Top-10~20，再让模型在候选池内做决策。

本模块实现两级打分融合：

    最终分 = 0.7 × 语义相似度(bge-m3 embedding cosine) + 0.3 × 关键词命中

- 语义分支：把工具的 name + description 向量化并缓存，用户 query 向量化后
  算余弦相似度。能覆盖"用户说'哪家好吃'，工具描述写'按分类查店铺'"
  这类字面不重合但语义相关的场景。
- 关键词分支：复用 INTENT_TOOL_MAP 的意图词表，保证"明确意图"被强命中，
  弥补 embedding 对短词、专有名词（KTV/SPA/兵马俑）的召回抖动。
- 兜底：embedding 不可用时自动降级为纯关键词；全部低于阈值时返回全量工具，
  宁可多给不可漏召回。

同时保留本地二次校验入口（validate_tool_calls），把 Schema 级参数校验
委托给 param_validator，依赖顺序校验仍在本模块内完成。

使用方式：
    from app.agent.tool_retriever import get_tool_retriever

    retriever = get_tool_retriever()

    filtered_tools = await retriever.aretrieve(user_query)   # 异步（推荐）
    errors = retriever.validate_tool_calls(response.tool_calls)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool

from app.config import settings

logger = logging.getLogger(__name__)


class ToolRetriever:
    """动态工具检索器。

    功能：
    1. 根据用户意图关键词动态检索相关工具（减少无关工具干扰）
    2. 本地约束解码：在工具执行前校验 tool_calls 合法性
    3. 保留兜底机制：检索结果为空时返回全部工具
    """

    # 关键词 → 工具映射（意图 → 应该使用的工具）
    INTENT_TOOL_MAP: Dict[str, List[str]] = {
        # 店铺搜索意图
        "店名|具体店铺|哪家|叫什么|老孙家|海底捞": [
            "search_shops_by_name", "get_shop_detail"
        ],
        # 分类推荐意图
        "推荐|附近|好吃|好玩|火锅|KTV|酒吧|美食|SPA|按摩|轰趴|美发|亲子|美甲|健身": [
            "list_shops_by_type", "get_shop_detail"
        ],
        # 优惠券意图
        "优惠|团购|折扣|券|划算|便宜|价格": [
            "list_shops_by_type", "list_vouchers", "search_shops_by_name"
        ],
        # 探店笔记意图
        "体验|评价|口碑|笔记|探店|怎么样|好不好|去过": [
            "search_blogs", "search_shops_by_name", "get_shop_detail"
        ],
        # 文旅知识意图
        "景点|攻略|门票|历史|文化|故事|开放|交通|行程|兵马俑|大雁塔|城墙|华清": [
            "search_knowledge"
        ],
        # 客服/反馈意图
        "投诉|举报|反馈|客服|人工|退款|建议|坑人|骗人": [
            "submit_feedback"
        ],
        # 天气查询意图
        "天气|下雨|下雪|冷|热|温度|气温|刮风|带伞|穿衣|防晒|雾霾|空气质量|多少度|冷不冷|热不热": [
            "query_weather"
        ],
        # 景点百科意图（比search_knowledge更具体的景点信息需求）
        "门票|开放时间|票价|多少钱|怎么去|怎么走|值得去|好玩吗|介绍|简介|游览|参观": [
            "query_attraction", "search_knowledge"
        ],
        # 交通出行意图
        "怎么去|怎么走|坐车|地铁|公交|驾车|步行|骑行|多远|多久|路线|导航|打车": [
            "query_transit", "query_attraction"
        ],
    }

    # 需要发送给LLM的基础工具描述时使用的字段映射
    TOOL_DESCRIPTIONS: Dict[str, str] = {
        "search_shops_by_name": "按店铺名称关键字搜索平台店铺（模糊匹配）",
        "list_shops_by_type": "按分类查店铺列表；传坐标时按距离排序",
        "get_shop_detail": "查询店铺详情（评分/人均/营业时间/地址等）",
        "list_vouchers": "实时查询店铺的优惠券（含库存/有效期）",
        "search_blogs": "搜索平台上的探店笔记/用户评价",
        "search_knowledge": "搜索城市文旅知识（景点/美食/攻略等）",
        "submit_feedback": "收集用户反馈/投诉/建议，记录到后台",
        "query_weather": "查询城市天气（实时天气+预报+穿衣建议）",
        "query_attraction": "查询景点百科信息（门票/开放时间/攻略/交通）",
        "query_transit": "查询两地交通出行方案（公交/地铁/驾车/步行/骑行）",
    }

    # 写操作工具（需要人工确认）
    WRITE_TOOLS: set = {"submit_feedback"}

    # 需要 shop_id 前置依赖的工具
    DEPENDENT_TOOLS: Dict[str, List[str]] = {
        "list_vouchers": ["search_shops_by_name", "list_shops_by_type", "get_shop_detail"],
        "get_shop_detail": ["search_shops_by_name", "list_shops_by_type"],
    }

    def __init__(self, all_tools: List[BaseTool]):
        """
        Args:
            all_tools: 所有可用工具列表
        """
        self.all_tools = all_tools
        self._tool_map: Dict[str, BaseTool] = {t.name: t for t in all_tools}
        # 工具向量索引（惰性构建，进程内缓存）
        self._tool_vectors: Optional[List[List[float]]] = None
        self._index_built_at: float = 0.0
        self._semantic_available: bool = False

    # ==================== 语义分支：工具向量索引 ====================

    def _tool_embedding_text(self, tool: BaseTool) -> str:
        """构造工具的向量化文本：name + 摘要 + 完整 description。"""
        summary = self.TOOL_DESCRIPTIONS.get(tool.name, "")
        full_desc = getattr(tool, "description", "") or ""
        return f"{tool.name}: {summary}\n{full_desc}"[:1500]

    def _build_index(self, force: bool = False) -> bool:
        """惰性构建工具向量索引。

        只在首次调用或缓存过期时真正调用 embedding 接口，之后走内存缓存。

        Returns:
            True 表示语义检索可用；False 表示需降级为纯关键词
        """
        ttl = getattr(settings, "tool_retrieval_cache_ttl", 86400)
        now = time.time()
        if not force and self._tool_vectors is not None and (now - self._index_built_at) < ttl:
            return self._semantic_available

        try:
            from app.services.embedding import get_embedding_client

            client = get_embedding_client()
            texts = [self._tool_embedding_text(t) for t in self.all_tools]
            vectors = client.embed_texts(texts)
            if len(vectors) != len(self.all_tools):
                logger.warning(
                    f"[ToolRetriever] 向量数量不匹配: {len(vectors)} vs {len(self.all_tools)}"
                )
                return False
            self._tool_vectors = vectors
            self._index_built_at = now
            self._semantic_available = True
            logger.info(f"[ToolRetriever] 工具向量索引构建完成: {len(vectors)} 个工具")
            return True
        except Exception as e:
            logger.warning(f"[ToolRetriever] 语义索引构建失败，降级为关键词检索: {e}")
            self._semantic_available = False
            return False

    def _semantic_scores(self, user_query: str) -> Optional[List[float]]:
        """计算用户 query 与每个工具的语义相似度（归一化到 0~1）。

        Returns:
            与 all_tools 等长的分数列表；不可用时返回 None
        """
        if not self._build_index():
            return None
        try:
            import numpy as np

            from app.services.embedding import get_embedding_client

            query_vec = get_embedding_client().embed_texts([user_query])[0]
            q = np.asarray(query_vec, dtype=np.float32)
            m = np.asarray(self._tool_vectors, dtype=np.float32)
            q_norm = float(np.linalg.norm(q)) or 1e-8
            m_norms = np.linalg.norm(m, axis=1)
            m_norms[m_norms == 0] = 1e-8
            cosine = (m @ q) / (m_norms * q_norm)
            # cosine ∈ [-1,1] → [0,1]，便于与关键词分加权
            return [float((c + 1.0) / 2.0) for c in cosine]
        except Exception as e:
            logger.warning(f"[ToolRetriever] 语义打分失败，降级为关键词: {e}")
            return None

    # ==================== 关键词分支 ====================

    def _keyword_scores(self, user_query: str) -> List[float]:
        """按意图词表计算每个工具的关键词命中分（归一化到 0~1）。"""
        hits: Dict[str, int] = {t.name: 0 for t in self.all_tools}
        for pattern, tool_names in self.INTENT_TOOL_MAP.items():
            keywords = pattern.split("|")
            matched = sum(1 for kw in keywords if kw and kw in user_query)
            if matched == 0:
                continue
            for name in tool_names:
                if name in hits:
                    hits[name] += matched

        max_hit = max(hits.values()) if hits else 0
        if max_hit == 0:
            return [0.0 for _ in self.all_tools]
        return [hits[t.name] / max_hit for t in self.all_tools]

    # ==================== 融合检索 ====================

    def retrieve(self, user_query: str) -> List[BaseTool]:
        """同步检索（测试/脚本用；服务内请用 aretrieve 避免阻塞事件循环）。

        Args:
            user_query: 用户查询文本

        Returns:
            筛选后的工具列表
        """
        return self._retrieve_impl(user_query, self._semantic_scores(user_query))

    async def aretrieve(self, user_query: str) -> List[BaseTool]:
        """异步检索：把阻塞的 embedding 调用放到线程池，避免阻塞事件循环。

        Args:
            user_query: 用户查询文本

        Returns:
            筛选后的工具列表
        """
        sem = await asyncio.to_thread(self._semantic_scores, user_query)
        return self._retrieve_impl(user_query, sem)

    def _retrieve_impl(
        self,
        user_query: str,
        semantic_scores: Optional[List[float]],
    ) -> List[BaseTool]:
        """混合打分 + Top-K 截断 + 兜底。

        流程：
        1. 语义分（不可用时置 0）与关键词分按权重融合
        2. 按融合分降序取 Top-K 候选池
        3. 丢弃低于 min_score 的候选
        4. 候选池为空 → 返回全部工具（宁可多给，不可漏召回）
        5. 最终按 max_tools_per_request 截断后绑定给模型

        Args:
            user_query: 用户查询文本
            semantic_scores: 语义分列表（None 表示不可用）

        Returns:
            筛选后的工具列表
        """
        if not settings.dynamic_tool_retrieval_enabled:
            return self.all_tools

        kw_scores = self._keyword_scores(user_query)
        w_sem = getattr(settings, "tool_retrieval_semantic_weight", 0.7)
        w_kw = getattr(settings, "tool_retrieval_keyword_weight", 0.3)

        # 语义不可用时，把权重全部让给关键词，避免分数整体塌陷
        if semantic_scores is None:
            w_sem, w_kw = 0.0, 1.0

        fused: List[Tuple[BaseTool, float, float, float]] = []
        for idx, tool in enumerate(self.all_tools):
            sem = semantic_scores[idx] if semantic_scores else 0.0
            kw = kw_scores[idx]
            score = w_sem * sem + w_kw * kw
            fused.append((tool, score, sem, kw))

        fused.sort(key=lambda x: x[1], reverse=True)

        top_k = getattr(settings, "tool_retrieval_top_k", 10)
        min_score = getattr(settings, "tool_retrieval_min_score", 0.15)
        max_bind = settings.max_tools_per_request

        candidates = [item for item in fused[:top_k] if item[1] >= min_score]

        if not candidates:
            logger.info(
                f"[ToolRetriever] 全部低于阈值 {min_score}，返回全量 {len(self.all_tools)} 个工具兜底"
            )
            return self.all_tools

        result = [item[0] for item in candidates[:max_bind]]

        logger.info(
            f"[ToolRetriever] query='{user_query[:40]}...' "
            f"语义={'ON' if semantic_scores else 'OFF'} "
            f"候选池={len(candidates)}/{len(self.all_tools)} "
            f"绑定={len(result)} → "
            f"{[(t.name, round(s, 3)) for t, s, _, _ in candidates[:max_bind]]}"
        )
        return result

    def validate_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[str]:
        """本地二次校验：Schema 级参数校验 + 工具名/调用顺序检查。

        三层校验（与 param_validator 配合）：
        1. 工具名合法性 —— 拦截模型幻想的工具
        2. Schema 级参数校验 —— 类型/枚举/必填/范围/长度，委托 param_validator
        3. 调用顺序依赖 —— list_vouchers 需要先有 shop_id 来源

        Args:
            tool_calls: LLM返回的tool_calls列表

        Returns:
            错误信息列表（空列表表示全部通过），可直接回传给模型修正
        """
        errors: List[str] = []

        if not tool_calls:
            return errors

        from app.agent.param_validator import get_param_validator

        validator = get_param_validator()

        # Schema 级参数校验（结构化错误，便于回传模型修正）
        for err in self.collect_param_errors(tool_calls):
            errors.append(err.to_llm_line(errors.__len__() + 1))

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {}) or {}

            if tool_name not in self._tool_map:
                errors.append(
                    f"❌ LLM返回了不存在的工具名「{tool_name}」，"
                    f"可用工具: {list(self._tool_map.keys())}"
                )
                logger.error(f"[ConstraintDecode] 非法工具名: {tool_name}")
                continue

            if tool_name in self.WRITE_TOOLS:
                logger.info(
                    f"[ConstraintDecode] 检测到写操作: {tool_name}, "
                    f"参数: {json.dumps(tool_args, ensure_ascii=False)[:100]}"
                )

        errors.extend(self._check_tool_order(tool_calls))

        if errors:
            logger.warning(f"[ConstraintDecode] 发现 {len(errors)} 个约束违规: {errors}")

        return errors

    def collect_param_errors(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """收集 Schema 级参数错误，并把可无损转换的参数就地规范化。

        校验通过时会把 `"108.94"` 这类字符串数字改写为 float 写回 tool_calls，
        确保工具执行时拿到的是正确类型（校验与规范化一次完成，避免二次遍历）。

        Args:
            tool_calls: LLM 返回的 tool_calls（通过时会被就地修正 args）

        Returns:
            ParamError 列表
        """
        from app.agent.param_validator import get_param_validator

        validator = get_param_validator()
        errors: List[Any] = []
        for tc in tool_calls or []:
            tool_name = tc.get("name", "")
            tool = self._tool_map.get(tool_name)
            if tool is None:
                continue
            schema = self._get_tool_schema(tool)
            if not schema:
                continue

            raw_args = tc.get("args") or {}
            errs, normalized = validator.validate_and_normalize(tool_name, raw_args, schema)
            errors.extend(errs)
            if not errs and normalized != raw_args:
                tc["args"] = normalized
                logger.debug(
                    f"[ConstraintDecode] 参数类型已规范化: {tool_name} "
                    f"{list(k for k in normalized if normalized[k] != raw_args.get(k))}"
                )
        return errors

    def _get_tool_schema(self, tool: BaseTool) -> Optional[Dict[str, Any]]:
        """获取工具的严格化 JSON Schema，并注入业务约束（枚举/范围）。

        LangChain 从 Python 类型注解生成的 Schema 只有基础类型，
        业务约束（分类白名单、西安坐标范围、店铺ID范围）原本在 ToolGuard
        里、工具函数执行时才检查。这里把它们前移到 Schema 层，使参数校验
        能在**执行前**完成，并把可选值一并回传给模型。

        Args:
            tool: LangChain 工具对象

        Returns:
            增强后的 Schema；失败返回 None
        """
        try:
            from app.agent.constrained_decoder import get_constrained_decoder

            raw = tool.get_input_jsonschema()
            schema = get_constrained_decoder().strictify_schema(raw)
            return self._inject_business_constraints(tool.name, schema)
        except Exception as e:
            logger.debug(f"[ToolRetriever] 获取工具 Schema 失败: {tool.name}: {e}")
            return None

    @staticmethod
    def _inject_business_constraints(
        tool_name: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把 ToolGuard 的业务白名单与值域注入 Schema。

        Args:
            tool_name: 工具名
            schema: 严格化后的 Schema（原地修改）

        Returns:
            增强后的 Schema
        """
        try:
            from app.agent.tool_guard import (
                VALID_SHOP_ID_RANGE,
                VALID_SHOP_TYPES,
                XI_AN_BOUNDS,
            )
        except Exception:
            return schema

        props = schema.get("properties", {})
        if not isinstance(props, dict):
            return schema

        if tool_name == "list_shops_by_type":
            if "type_name" in props:
                props["type_name"]["enum"] = sorted(VALID_SHOP_TYPES)
            for axis, (lo_key, hi_key) in (("x", ("x_min", "x_max")), ("y", ("y_min", "y_max"))):
                if axis in props:
                    props[axis]["minimum"] = XI_AN_BOUNDS[lo_key]
                    props[axis]["maximum"] = XI_AN_BOUNDS[hi_key]

        if tool_name in ("get_shop_detail", "list_vouchers") and "shop_id" in props:
            props["shop_id"]["minimum"] = VALID_SHOP_ID_RANGE[0]
            props["shop_id"]["maximum"] = VALID_SHOP_ID_RANGE[1]

        return schema

    def _get_required_params(self, tool: BaseTool) -> List[str]:
        """从工具的 JSON Schema 中提取必填参数列表。"""
        try:
            schema = tool.get_input_jsonschema()
            return schema.get("required", [])
        except Exception:
            return []

    def _check_tool_order(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """检查工具调用顺序是否合理（依赖关系）。"""
        errors = []

        tools_called = [tc["name"] for tc in tool_calls]

        for tc in tool_calls:
            tool_name = tc["name"]
            if tool_name in self.DEPENDENT_TOOLS:
                prerequisites = self.DEPENDENT_TOOLS[tool_name]
                # 检查本轮或历史中是否有前置工具
                has_prereq = any(p in tools_called for p in prerequisites)
                if not has_prereq:
                    errors.append(
                        f"⚠️ 工具「{tool_name}」依赖前置工具 {prerequisites}，"
                        f"但本轮未检测到前置调用（可能shop_id为编造）"
                    )
                    logger.warning(
                        f"[ConstraintDecode] 工具顺序违规: {tool_name} "
                        f"缺少前置 {prerequisites}"
                    )

        return errors

    def get_tool_descriptions(self, tool_names: List[str]) -> str:
        """生成工具描述摘要（用于日志和调试）。

        Args:
            tool_names: 工具名列表

        Returns:
            格式化的工具描述字符串
        """
        lines = []
        for name in tool_names:
            desc = self.TOOL_DESCRIPTIONS.get(name, name)
            write_tag = " [写操作]" if name in self.WRITE_TOOLS else ""
            lines.append(f"  - {name}: {desc}{write_tag}")
        return "\n".join(lines)


# 获取所有工具列表的辅助函数（延迟导入避免循环依赖）
def _get_all_tools() -> List[BaseTool]:
    from app.agent.tools import ALL_TOOLS
    return ALL_TOOLS


# 全局检索器实例（懒加载）
_retriever: Optional[ToolRetriever] = None


def get_tool_retriever() -> ToolRetriever:
    """获取全局 ToolRetriever 实例（懒加载单例）。"""
    global _retriever
    if _retriever is None:
        _retriever = ToolRetriever(_get_all_tools())
    return _retriever