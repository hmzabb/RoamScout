"""防御体系V3.0 测试套件：测试全部6层防御是否正常工作。

测试覆盖：
- 第一层：工具描述优化（通过实际调用验证）
- 第二层：参数校验守卫（ToolGuard）
- 第三层：调用审计日志（ToolCallAuditor）
- 第四层：结果质量检查（ResultQualityChecker）
- 第五层：System Prompt强化（通过Agent行为验证）
- 第六层：结构化思考（StructuredThinking）

运行方式：
    cd chang_an_ai
    python -m pytest tests/test_defense_layers.py -v
    # 或
    python tests/test_defense_layers.py
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 第二层：ToolGuard 测试 ====================

class TestToolGuard:
    """第二层防御：参数校验守卫测试。"""

    def setup_method(self):
        """每个测试前重置状态。"""
        from app.agent.tool_guard import ToolGuard
        ToolGuard.clear_duplicate_cache()

    def test_validate_shop_type_valid(self):
        """测试合法分类名。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_shop_type("美食")
        assert result is None, f"合法分类名不应被拒绝: {result}"

    def test_validate_shop_type_invalid(self):
        """测试非法分类名。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_shop_type("编程")
        assert result is not None
        assert "不合法" in result or "不在" in result

    def test_validate_shop_type_empty(self):
        """测试空分类名。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_shop_type("")
        assert result is not None
        assert "不能为空" in result

    def test_validate_coordinates_valid_xian(self):
        """测试西安坐标范围。"""
        from app.agent.tool_guard import ToolGuard
        # 钟楼坐标
        result = ToolGuard.validate_coordinates(108.9425, 34.2610)
        assert result is None, f"西安坐标不应被拒绝: {result}"

    def test_validate_coordinates_out_of_range(self):
        """测试超出西安范围的坐标。"""
        from app.agent.tool_guard import ToolGuard
        # 北京坐标
        result = ToolGuard.validate_coordinates(116.4074, 39.9042)
        assert result is not None
        assert "不在西安范围" in result

    def test_validate_shop_id_valid(self):
        """测试合法shop_id。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_shop_id(12345)
        assert result is None

    def test_validate_shop_id_invalid_range(self):
        """测试超出范围的shop_id。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_shop_id(999999)
        assert result is not None
        assert "不在合理范围" in result

    def test_validate_shop_id_none(self):
        """测试空shop_id。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_shop_id(None)
        assert result is not None
        assert "不能为空" in result

    def test_validate_search_query_valid(self):
        """测试合法搜索词。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_search_query("大雁塔")
        assert result is None

    def test_validate_search_query_empty(self):
        """测试空搜索词。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.validate_search_query("")
        assert result is not None
        assert "不能为空" in result

    def test_check_duplicate_call_first_time(self):
        """测试首次调用不被拦截。"""
        from app.agent.tool_guard import ToolGuard
        result = ToolGuard.check_duplicate_call("test_tool", {"param": "value"})
        assert result is None, "首次调用不应被拦截"

    def test_check_duplicate_call_duplicate(self):
        """测试重复调用被拦截。"""
        from app.agent.tool_guard import ToolGuard
        import time

        # 第一次调用
        ToolGuard.check_duplicate_call("test_tool", {"param": "value"})

        # 立即第二次调用（应被拦截）
        result = ToolGuard.check_duplicate_call("test_tool", {"param": "value"})
        assert result is not None
        assert "重复调用" in result


# ==================== 第三层：ToolCallAuditor 测试 ====================

class TestToolCallAuditor:
    """第三层防御：调用审计日志测试。"""

    def setup_method(self):
        """每个测试前重置统计。"""
        from app.agent.graph import ToolCallAuditor
        ToolCallAuditor.reset_stats()

    def test_audit_missing_voucher_tool(self):
        """测试用户提到优惠但未调list_vouchers。"""
        from app.agent.graph import ToolCallAuditor

        tool_calls = [
            {"name": "list_shops_by_type", "args": {"type_name": "美食"}}
        ]
        warnings = ToolCallAuditor.audit_tool_calls(
            user_query="钟楼附近有优惠的火锅店",
            tool_calls=tool_calls
        )

        assert len(warnings) > 0
        assert any("优惠" in w and "list_vouchers" in w for w in warnings)

    def test_audit_direct_voucher_call(self):
        """测试直接调list_vouchers但未先获取shop_id。"""
        from app.agent.graph import ToolCallAuditor

        tool_calls = [
            {"name": "list_vouchers", "args": {"shop_id": 123}}
        ]
        warnings = ToolCallAuditor.audit_tool_calls(
            user_query="查一下123号店的优惠券",
            tool_calls=tool_calls
        )

        assert len(warnings) > 0
        assert any("编造" in w or "直接调" in w for w in warnings)

    def test_audit_duplicate_tool_calls(self):
        """测试同一工具相同参数重复调用。"""
        from app.agent.graph import ToolCallAuditor

        tool_calls = [
            {"name": "search_shops_by_name", "args": {"name": "泡馍"}},
            {"name": "search_shops_by_name", "args": {"name": "泡馍"}},
        ]
        warnings = ToolCallAuditor.audit_tool_calls(
            user_query="找泡馍店",
            tool_calls=tool_calls
        )

        assert len(warnings) > 0
        assert any("重复调用" in w for w in warnings)

    def test_audit_normal_call_no_warning(self):
        """测试正常调用无警告。"""
        from app.agent.graph import ToolCallAuditor

        tool_calls = [
            {"name": "list_shops_by_type", "args": {"type_name": "美食"}},
            {"name": "search_knowledge", "args": {"query": "大雁塔"}},
        ]
        warnings = ToolCallAuditor.audit_tool_calls(
            user_query="推荐钟楼附近的美食和大雁塔攻略",
            tool_calls=tool_calls
        )

        # 正常调用可能有轻微警告，但不应该有严重警告
        critical_warnings = [w for w in warnings if "编造" in w or "直接调" in w]
        assert len(critical_warnings) == 0, f"不应有严重警告: {critical_warnings}"

    def test_get_stats(self):
        """测试获取审计统计。"""
        from app.agent.graph import ToolCallAuditor

        # 执行几次审计
        ToolCallAuditor.audit_tool_calls(
            user_query="测试",
            tool_calls=[{"name": "search_knowledge", "args": {"query": "test"}}]
        )

        stats = ToolCallAuditor.get_stats()
        assert stats["total_calls"] >= 1
        assert "tool_usage_count" in stats
        assert "anomalies_detected" in stats


# ==================== 第四层：ResultQualityChecker 测试 ====================

class TestResultQualityChecker:
    """第四层防御：结果质量检查测试。"""

    def test_empty_result_detection_shops(self):
        """检测空店铺结果。"""
        from app.agent.parallel_tools import ResultQualityChecker

        empty_result = json.dumps({"count": 0, "shops": []})
        enhanced = ResultQualityChecker.check_and_enhance_result(
            tool_name="search_shops_by_name",
            result_content=empty_result,
            tool_args={"name": "不存在的店"}
        )

        data = json.loads(enhanced)
        assert "_quality_check" in data
        assert data["_quality_check"]["status"] == "empty"
        assert "提示" in data["_quality_check"]["hint"]

    def test_empty_result_detection_vouchers(self):
        """检测空优惠券结果。"""
        from app.agent.parallel_tools import ResultQualityChecker

        empty_result = json.dumps({"count": 0, "vouchers": []})
        enhanced = ResultQualityChecker.check_and_enhance_result(
            tool_name="list_vouchers",
            result_content=empty_result,
            tool_args={"shop_id": 123}
        )

        data = json.loads(enhanced)
        assert "_quality_check" in data
        assert data["_quality_check"]["status"] == "empty"

    def test_error_result_enhancement(self):
        """错误结果增强。"""
        from app.agent.parallel_tools import ResultQualityChecker

        error_result = json.dumps({"error": "没有找到分类「编程」，可用分类: [...]"})
        enhanced = ResultQualityChecker.check_and_enhance_result(
            tool_name="list_shops_by_type",
            result_content=error_result,
            tool_args={"type_name": "编程"}
        )

        data = json.loads(enhanced)
        assert "_quality_check" in data
        assert data["_quality_check"]["status"] == "error"
        assert "suggestion" in data["_quality_check"]

    def test_data_quality_validation_abnormal_score(self):
        """数据质量检查：异常评分。"""
        from app.agent.parallel_tools import ResultQualityChecker

        abnormal_data = {
            "count": 1,
            "shops": [{
                "id": 1,
                "name": "测试店",
                "score": 10.0,  # 异常：超过5分
                "avg_price": 100,
            }]
        }
        result_str = json.dumps(abnormal_data)
        enhanced = ResultQualityChecker.check_and_enhance_result(
            tool_name="list_shops_by_type",
            result_content=result_str,
            tool_args={"type_name": "美食"}
        )

        data = json.loads(enhanced)
        assert "_quality_check" in data
        assert data["_quality_check"]["status"] == "warnings"
        assert any("评分异常" in w for w in data["_quality_check"]["warnings"])

    def test_normal_result_unchanged(self):
        """正常结果不应被修改。"""
        from app.agent.parallel_tools import ResultQualityChecker

        normal_data = {
            "count": 2,
            "shops": [
                {"id": 1, "name": "店A", "score": 4.5, "avg_price": 80},
                {"id": 2, "name": "店B", "score": 4.8, "avg_price": 120},
            ]
        }
        result_str = json.dumps(normal_data)
        enhanced = ResultQualityChecker.check_and_enhance_result(
            tool_name="list_shops_by_type",
            result_content=result_str,
            tool_args={"type_name": "美食"}
        )

        data = json.loads(enhanced)
        # 正常数据可能没有 _quality_check，或者只有基本信息
        if "_quality_check" in data:
            assert data["_quality_check"]["status"] != "error"
            assert data["_quality_check"]["status"] != "empty"


# ==================== 第六层：StructuredThinking 测试 ====================

class TestStructuredThinking:
    """第六层防御：结构化思考测试。"""

    def setup_method(self):
        """每个测试前后重置状态。"""
        from app.agent.structured_thinking import StructuredThinking
        StructuredThinking.disable()
        StructuredThinking.reset_stats()

    def test_enable_disable(self):
        """测试启用/禁用。"""
        from app.agent.structured_thinking import StructuredThinking

        assert not StructuredThinking.is_enabled()
        StructuredThinking.enable()
        assert StructuredThinking.is_enabled()
        StructuredThinking.disable()
        assert not StructuredThinking.is_enabled()

    def test_agent_think_model_validation(self):
        """测试AgentThink模型验证。"""
        from app.agent.structured_thinking import AgentThink

        # 有效输入
        think = AgentThink(
            user_intent="用户想找钟楼附近的火锅店",
            needed_tools=["list_shops_by_type"],
            tool_reasoning={"list_shops_by_type": "用户要找火锅，需要按分类搜索"},
            is_travel_related=True,
            confidence=0.9,
            potential_issues=["可能返回空结果"]
        )
        assert think.is_valid

        # 无效：不相关
        think_invalid = AgentThink(
            user_intent="用户想学Python",
            needed_tools=[],
            tool_reasoning={},
            is_travel_related=False,
            confidence=0.8,
            potential_issues=[]
        )
        assert not think_invalid.is_valid

    def test_agent_think_invalid_tool_name(self):
        """测试无效工具名。"""
        from app.agent.structured_thinking import AgentThink
        import pytest as pt

        with pt.raises(Exception):  # Pydantic验证异常
            AgentThink(
                user_intent="测试",
                needed_tools=["invalid_tool"],  # 无效工具名
                tool_reasoning={},
                is_travel_related=True,
                confidence=0.9
            )

    def test_to_prompt_hint(self):
        """测试生成提示信息。"""
        from app.agent.structured_thinking import AgentThink

        think = AgentThink(
            user_intent="帮我找附近的火锅店",
            needed_tools=["list_shops_by_type"],
            tool_reasoning={"list_shops_by_type": "按分类搜索"},
            is_travel_related=True,
            confidence=0.95
        )

        hint = think.to_prompt_hint()
        assert "意图分析" in hint
        assert "计划调用工具" in hint
        assert "list_shops_by_type" in hint

    def test_extract_json_from_text(self):
        """测试从文本中提取JSON。"""
        from app.agent.structured_thinking import StructuredThinking

        # markdown代码块中的JSON
        text = '```json\n{"key": "value"}\n```'
        result = StructuredThinking._extract_json(text)
        assert result == '{"key": "value"}'

        # 纯JSON
        text2 = '{"a": 1}'
        result2 = StructuredThinking._extract_json(text2)
        assert result2 == '{"a": 1}'

        # 无法提取
        text3 = "这是一段普通文本"
        result3 = StructuredThinking._extract_json(text3)
        assert result3 is None

    def test_get_stats(self):
        """测试获取统计信息。"""
        from app.agent.structured_thinking import StructuredThinking

        stats = StructuredThinking.get_stats()
        assert "enabled" in stats
        assert "total_thinks" in stats
        assert "valid_thinks" in stats


# ==================== 集成测试 ====================

class TestDefenseIntegration:
    """防御体系集成测试。"""

    def test_full_defense_chain_mock(self):
        """使用Mock测试完整防御链路。"""
        # 这个测试模拟完整的请求流程
        # 由于需要真实的LLM和Java服务，这里只做结构验证

        # 验证所有模块可以导入
        from app.agent.tool_guard import ToolGuard
        from app.agent.graph import ToolCallAuditor
        from app.agent.parallel_tools import ResultQualityChecker, ParallelToolNode
        from app.agent.structured_thinking import StructuredThinking, AgentThink
        from app.prompts.agent import AGENT_SYSTEM
        from app.agent.tools import ALL_TOOLS

        # 验证工具列表完整
        assert len(ALL_TOOLS) == 7
        tool_names = [t.name for t in ALL_TOOLS]
        assert "search_shops_by_name" in tool_names
        assert "list_shops_by_type" in tool_names
        assert "get_shop_detail" in tool_names
        assert "list_vouchers" in tool_names
        assert "search_blogs" in tool_names
        assert "search_knowledge" in tool_names
        assert "submit_feedback" in tool_names

        # 验证System Prompt包含关键规则
        assert "决策树" in AGENT_SYSTEM
        assert "防幻觉" in AGENT_SYSTEM
        assert "死循环" in AGENT_SYSTEM
        assert "禁止编造" in AGENT_SYSTEM

        print("\n✅ 所有防御模块导入成功！")
        print(f"✅ 工具数量: {len(ALL_TOOLS)}")
        print(f"✅ System Prompt长度: {len(AGENT_SYSTEM)} 字符")


# ==================== 第二层（调用链路层）：约束解码 / 参数校验 / 工具检索 ====================

class TestConstrainedDecoder:
    """第二层·防护一：约束解码测试。

    云 API 无 logits 访问，因此验证的是等价实现：
    Schema 封闭化 + 响应侧多轮修复 + 结构化报错。
    """

    def test_strictify_closes_object(self):
        """Schema 严格化应封闭对象、禁止未声明字段。"""
        from app.agent.constrained_decoder import get_constrained_decoder
        from app.agent.tools import search_shops_by_name

        schema = get_constrained_decoder().strictify_schema(
            search_shops_by_name.get_input_jsonschema()
        )
        assert schema["additionalProperties"] is False

    def test_strictify_keeps_optional_params_optional(self):
        """可选参数不应被强行改成必填（否则模型会编造值）。"""
        from app.agent.constrained_decoder import get_constrained_decoder
        from app.agent.tools import list_shops_by_type

        schema = get_constrained_decoder().strictify_schema(
            list_shops_by_type.get_input_jsonschema()
        )
        props = set(schema["properties"].keys())
        required = set(schema.get("required", []))
        assert required.issubset(props)
        assert required != props, "x/y 为可选参数，不应全部必填"

    def test_strictify_force_all_required(self):
        """显式要求全量必填时，required 应覆盖全部属性。"""
        from app.agent.constrained_decoder import get_constrained_decoder
        from app.agent.tools import list_shops_by_type

        schema = get_constrained_decoder().strictify_schema(
            list_shops_by_type.get_input_jsonschema(),
            force_all_required=True,
        )
        assert set(schema["required"]) == set(schema["properties"].keys())

    def test_decode_plain_json(self):
        """纯 JSON 应直接解析成功。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        r = get_constrained_decoder().decode('{"name": "老孙家"}')
        assert r.ok is True
        assert r.data == {"name": "老孙家"}

    def test_decode_repairs_markdown_fence(self):
        """markdown 围栏包裹的 JSON 应被修复。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        raw = '```json\n{"name": "老孙家"}\n```'
        r = get_constrained_decoder().decode(raw)
        assert r.ok is True
        assert r.repaired is True
        assert "移除markdown围栏" in r.repair_log

    def test_decode_repairs_trailing_comma(self):
        """尾逗号应被移除。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        r = get_constrained_decoder().decode('{"name": "老孙家",}')
        assert r.ok is True
        assert r.data == {"name": "老孙家"}

    def test_decode_extracts_json_from_prose(self):
        """前后带解释文字时应截取 JSON 片段。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        r = get_constrained_decoder().decode('好的，我来查：{"name": "泡馍"} 请稍等')
        assert r.ok is True
        assert r.data == {"name": "泡馍"}

    def test_decode_repairs_truncated_json(self):
        """截断的 JSON 应补全括号。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        r = get_constrained_decoder().decode('{"tool_calls": [{"name": "list_shops_by_type", "args": {"type_name": "美食"}')
        assert r.ok is True
        assert "补全未闭合括号" in r.repair_log

    def test_decode_returns_structured_error(self):
        """完全非 JSON 时应返回结构化报错而非抛异常。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        r = get_constrained_decoder().decode("我无法回答这个问题")
        assert r.ok is False
        assert r.error
        assert "JSON" in r.error

    def test_is_token_level_false_for_cloud_api(self):
        """云 API 后端不具备 token 级屏蔽能力。"""
        from app.agent.constrained_decoder import get_constrained_decoder

        d = get_constrained_decoder()
        assert d.is_token_level is False


class TestParamValidator:
    """第二层·防护二：参数校验测试。"""

    def _schema(self, tool_name):
        from app.agent.tool_retriever import get_tool_retriever

        retriever = get_tool_retriever()
        return retriever._get_tool_schema(retriever._tool_map[tool_name])

    def test_valid_args_pass(self):
        """合法参数应零错误。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "search_knowledge", {"query": "大雁塔历史"}, self._schema("search_knowledge")
        )
        assert errs == []

    def test_missing_required(self):
        """缺少必填参数应报 MISSING_REQUIRED。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "search_shops_by_name", {}, self._schema("search_shops_by_name")
        )
        assert any(e.code == "MISSING_REQUIRED" and e.param == "name" for e in errs)

    def test_type_mismatch(self):
        """类型不符且无法转换时报 TYPE_MISMATCH。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "search_shops_by_name", {"name": {"a": 1}}, self._schema("search_shops_by_name")
        )
        assert any(e.code == "TYPE_MISMATCH" for e in errs)

    def test_enum_mismatch_with_suggestion(self):
        """非法枚举应报 ENUM_MISMATCH，并给出近似建议。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "list_shops_by_type", {"type_name": "美食类"}, self._schema("list_shops_by_type")
        )
        enum_errs = [e for e in errs if e.code == "ENUM_MISMATCH"]
        assert enum_errs
        assert "美食" in enum_errs[0].message

    def test_out_of_range_coordinate(self):
        """越界坐标应报 OUT_OF_RANGE。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "list_shops_by_type",
            {"type_name": "美食", "x": 120.0},
            self._schema("list_shops_by_type"),
        )
        assert any(e.code == "OUT_OF_RANGE" and e.param == "x" for e in errs)

    def test_unknown_param_rejected(self):
        """模型臆造的字段应被拦截。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "search_knowledge",
            {"query": "大雁塔", "hack": 1},
            self._schema("search_knowledge"),
        )
        assert any(e.code == "UNKNOWN_PARAM" and e.param == "hack" for e in errs)

    def test_string_number_normalized(self):
        """字符串数字应被无损转换为 float 并回写。"""
        from app.agent.param_validator import get_param_validator

        errs, normalized = get_param_validator().validate_and_normalize(
            "list_shops_by_type",
            {"type_name": "美食", "x": "108.94", "y": "34.26"},
            self._schema("list_shops_by_type"),
        )
        assert errs == []
        assert isinstance(normalized["x"], float)
        assert isinstance(normalized["y"], float)

    def test_optional_anyof_type_resolved(self):
        """Optional[float] 的 anyOf 结构应被展开，否则校验会被跳过。"""
        from app.agent.param_validator import get_param_validator

        resolved = get_param_validator()._resolve_prop(
            {"anyOf": [{"type": "number"}, {"type": "null"}]}
        )
        assert resolved.get("type") == "number"

    def test_format_for_llm_readable(self):
        """报错应能格式化为模型可读的修正指令。"""
        from app.agent.param_validator import get_param_validator

        errs = get_param_validator().validate(
            "list_shops_by_type", {"type_name": "火锅"}, self._schema("list_shops_by_type")
        )
        text = get_param_validator().format_for_llm(errs)
        assert "校验失败" in text
        assert "ENUM_MISMATCH" in text


class TestToolRetriever:
    """第二层·防护三：工具检索测试。"""

    def test_keyword_scores_hit(self):
        """明确意图词应命中对应工具。"""
        from app.agent.tool_retriever import ToolRetriever
        from app.agent.tools import ALL_TOOLS

        r = ToolRetriever(ALL_TOOLS)
        scores = r._keyword_scores("推荐几个好吃的火锅店")
        pairs = dict(zip([t.name for t in ALL_TOOLS], scores))
        assert pairs["list_shops_by_type"] > 0

    def test_topk_candidate_pool(self):
        """检索结果应收缩为候选池而非全量。"""
        from app.agent.tool_retriever import ToolRetriever
        from app.agent.tools import ALL_TOOLS

        r = ToolRetriever(ALL_TOOLS)
        picked = r._retrieve_impl("推荐几个好吃的火锅店", None)
        assert 0 < len(picked) < len(ALL_TOOLS)
        assert any(t.name == "list_shops_by_type" for t in picked)

    def test_semantic_unavailable_degrades_to_keyword(self):
        """embedding 不可用时应降级为纯关键词，不抛异常。"""
        import app.services.embedding as emb_mod
        from app.agent.tool_retriever import ToolRetriever
        from app.agent.tools import ALL_TOOLS

        class BrokenClient:
            def embed_texts(self, texts):
                raise RuntimeError("network unreachable")

        original = emb_mod.get_embedding_client
        emb_mod.get_embedding_client = lambda: BrokenClient()
        try:
            r = ToolRetriever(ALL_TOOLS)
            sem = r._semantic_scores("推荐几个好吃的火锅店")
            assert sem is None
            assert r._semantic_available is False
            picked = r._retrieve_impl("推荐几个好吃的火锅店", sem)
            assert len(picked) > 0
        finally:
            emb_mod.get_embedding_client = original

    def test_semantic_index_cached(self):
        """工具向量索引应缓存，不重复请求 embedding。"""
        import numpy as np

        import app.services.embedding as emb_mod
        from app.agent.tool_retriever import ToolRetriever
        from app.agent.tools import ALL_TOOLS

        counter = {"n": 0}

        class FakeClient:
            def embed_texts(self, texts):
                counter["n"] += 1
                return [np.ones(8, dtype=np.float32).tolist() for _ in texts]

        original = emb_mod.get_embedding_client
        emb_mod.get_embedding_client = lambda: FakeClient()
        try:
            r = ToolRetriever(ALL_TOOLS)
            r._semantic_scores("测试一")
            built = counter["n"]
            r._semantic_scores("测试二")
            assert counter["n"] == built + 1, "第二次应只向量化 query，不重建索引"
        finally:
            emb_mod.get_embedding_client = original

    def test_empty_candidates_fallback_to_all(self):
        """全部低于阈值时应返回全量工具兜底。"""
        from app.agent.tool_retriever import ToolRetriever
        from app.agent.tools import ALL_TOOLS

        r = ToolRetriever(ALL_TOOLS)
        picked = r._retrieve_impl("zzzz", None)
        assert len(picked) == len(ALL_TOOLS)

    def test_validate_blocks_unknown_tool(self):
        """不存在的工具名应被拦截。"""
        from app.agent.tool_retriever import get_tool_retriever

        errors = get_tool_retriever().validate_tool_calls(
            [{"name": "fake_tool", "args": {}}]
        )
        assert any("fake_tool" in e for e in errors)

    def test_validate_blocks_bad_enum(self):
        """非法枚举应被拦截，报错含可选值。"""
        from app.agent.tool_retriever import get_tool_retriever

        errors = get_tool_retriever().validate_tool_calls(
            [{"name": "list_shops_by_type", "args": {"type_name": "火锅"}}]
        )
        assert errors
        assert "ENUM_MISMATCH" in errors[0]

    def test_validate_passes_and_normalizes(self):
        """合法调用应通过，并把字符串数字规范化为 float。"""
        from app.agent.tool_retriever import get_tool_retriever

        calls = [{
            "name": "list_shops_by_type",
            "args": {"type_name": "美食", "x": "108.94", "y": "34.26"},
            "id": "c1",
        }]
        errors = get_tool_retriever().validate_tool_calls(calls)
        assert errors == []
        assert isinstance(calls[0]["args"]["x"], float)

    def test_retrieve_disabled_returns_all(self):
        """关闭动态检索时应返回全量工具。"""
        from app.agent.tool_retriever import ToolRetriever
        from app.agent.tools import ALL_TOOLS
        from app.config import settings

        r = ToolRetriever(ALL_TOOLS)
        original = settings.dynamic_tool_retrieval_enabled
        settings.dynamic_tool_retrieval_enabled = False
        try:
            assert len(r._retrieve_impl("推荐火锅", None)) == len(ALL_TOOLS)
        finally:
            settings.dynamic_tool_retrieval_enabled = original


class TestParamRetryLoop:
    """第二层闭环：校验失败 → 回传报错 → 模型修正。"""

    def test_graph_has_retry_edge(self):
        """图中应存在 agent → agent 的参数修正自环。"""
        from app.agent.graph import build_agent

        graph = build_agent().get_graph()
        edges = [(e.source, e.target) for e in graph.edges]
        assert ("agent", "agent") in edges
        assert ("agent", "tools") in edges
        assert ("tools", "agent") in edges

    def test_state_has_retry_fields(self):
        """状态应含参数修正计数字段。"""
        from app.agent.state import AgentState

        assert "param_retry" in AgentState.__annotations__
        assert "need_param_retry" in AgentState.__annotations__

    def test_retry_loop_repairs_bad_enum(self):
        """模型第 1 次给非法枚举，收到报错后第 2 次给合法答案。"""
        import asyncio

        from langchain_core.messages import AIMessage, HumanMessage

        import app.agent.graph as graph_mod

        class FakeModel:
            def __init__(self, *a, **kw):
                self.calls = 0

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "list_shops_by_type",
                            "args": {"type_name": "火锅"},
                            "id": "call_bad",
                            "type": "tool_call",
                        }],
                    )
                return AIMessage(content="已为你找到美食类店铺。", tool_calls=[])

        fake = FakeModel()
        original = graph_mod.ChatOpenAI
        graph_mod.ChatOpenAI = lambda *a, **kw: fake
        try:
            agent = graph_mod.build_agent()
            result = asyncio.run(agent.ainvoke(
                {
                    "messages": [HumanMessage(content="推荐几个好吃的火锅店")],
                    "iter": 0,
                    "param_retry": 0,
                    "need_param_retry": False,
                },
                config={"recursion_limit": 30},
            ))
        finally:
            graph_mod.ChatOpenAI = original

        assert fake.calls == 2, "应发生一次修正重生成"
        tool_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
        assert tool_msgs, "校验失败应回传 ToolMessage"
        assert "ENUM_MISMATCH" in tool_msgs[0].content
        assert result.get("param_retry") == 0
        assert result.get("need_param_retry") is False

    def test_retry_limit_stops_loop(self):
        """模型始终给非法参数时，应在重试上限后终止并兜底。"""
        import asyncio

        from langchain_core.messages import AIMessage, HumanMessage

        import app.agent.graph as graph_mod
        from app.config import settings

        class AlwaysBadModel:
            def __init__(self, *a, **kw):
                self.calls = 0

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                self.calls += 1
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "list_shops_by_type",
                        "args": {"type_name": "火锅"},
                        "id": f"bad_{self.calls}",
                        "type": "tool_call",
                    }],
                )

        fake = AlwaysBadModel()
        original = graph_mod.ChatOpenAI
        graph_mod.ChatOpenAI = lambda *a, **kw: fake
        try:
            agent = graph_mod.build_agent()
            result = asyncio.run(agent.ainvoke(
                {
                    "messages": [HumanMessage(content="推荐几个好吃的火锅店")],
                    "iter": 0,
                    "param_retry": 0,
                    "need_param_retry": False,
                },
                config={"recursion_limit": 30},
            ))
        finally:
            graph_mod.ChatOpenAI = original

        assert fake.calls <= settings.param_validation_max_retry + 1
        assert "没能正确组装查询参数" in result["messages"][-1].content


# ==================== 第三层：权限分级 / 幂等键 / 超时分级 ====================

class TestPermissionTier:
    """第三层安全治理：权限分级（模型不能决定"能不能做"）。"""

    def test_read_tool_permission(self):
        """只读工具应被判定为 READ 权限。"""
        from app.agent.policy_engine import get_tool_permission, ToolPermission

        for name in [
            "search_shops_by_name", "list_shops_by_type", "get_shop_detail",
            "list_vouchers", "search_blogs", "search_knowledge",
        ]:
            assert get_tool_permission(name) == ToolPermission.READ

    def test_write_tool_permission(self):
        """写工具应被判定为 WRITE 权限。"""
        from app.agent.policy_engine import get_tool_permission, ToolPermission

        assert get_tool_permission("submit_feedback") == ToolPermission.WRITE

    def test_unregistered_tool_is_most_conservative(self):
        """未登记工具按最保守的 MONEY 处理，不允许模型自由调用。"""
        from app.agent.policy_engine import get_tool_permission, ToolPermission

        assert get_tool_permission("delete_database") == ToolPermission.MONEY
        assert get_tool_permission("create_order") == ToolPermission.MONEY

    def test_whitelist_rule_denies_unknown_tool(self):
        """白名单规则应拒绝未登记的工具。"""
        import asyncio
        from app.agent.policy_engine import ToolWhitelistRule

        rule = ToolWhitelistRule()
        decision = asyncio.run(rule.evaluate("hack_tool", {}, {}))
        assert not decision.allowed

    def test_whitelist_rule_allows_registered_tool(self):
        """白名单规则应放行已登记工具。"""
        import asyncio
        from app.agent.policy_engine import ToolWhitelistRule

        rule = ToolWhitelistRule()
        decision = asyncio.run(rule.evaluate("search_knowledge", {"query": "兵马俑"}, {}))
        assert decision.allowed

    def test_read_tool_auto_execute(self):
        """只读工具应自动放行（不进入确认流程）。"""
        import asyncio
        from app.agent.policy_engine import PermissionTierRule

        rule = PermissionTierRule()
        ctx = {}
        decision = asyncio.run(rule.evaluate("search_knowledge", {"query": "x"}, ctx))
        assert decision.allowed
        assert ctx.get("_permission") == "read"

    def test_write_tool_requires_confirm(self):
        """写操作必须进入人工确认流程。"""
        import asyncio
        from app.agent.policy_engine import WriteConfirmRule

        rule = WriteConfirmRule()
        decision = asyncio.run(rule.evaluate(
            "submit_feedback",
            {"feedback_type": "complaint", "content": "服务态度差"},
            {"session_id": "sess_test"},
        ))
        assert decision.requires_confirmation is True
        assert decision.confirm_token


class TestIdempotencyRequestId:
    """第三层安全治理：写操作必须携带 request_id，下游据此去重。"""

    def test_request_id_injected_into_write_args(self):
        """写操作的 request_id 必须被注入到 tool_args，下游才能去重。"""
        import asyncio
        from app.agent.policy_engine import IdempotencyRule

        rule = IdempotencyRule()
        args = {"feedback_type": "complaint", "content": "测试"}
        ctx = {"session_id": "sess_abc"}
        asyncio.run(rule.evaluate("submit_feedback", args, ctx))

        assert "request_id" in args
        assert args["request_id"].startswith("req_")
        assert ctx["_request_id"] == args["request_id"]

    def test_request_id_stable_for_same_operation(self):
        """同一会话+同一工具+同一参数的 request_id 必须稳定（重试可复现）。"""
        from app.agent.policy_engine import IdempotencyRule

        a = IdempotencyRule.make_request_id(
            "submit_feedback", {"content": "内容", "feedback_type": "bug"}, "sess_1"
        )
        b = IdempotencyRule.make_request_id(
            "submit_feedback", {"feedback_type": "bug", "content": "内容"}, "sess_1"
        )
        assert a == b, "参数顺序不同但语义相同，request_id 应保持一致"

    def test_request_id_differs_across_sessions(self):
        """不同会话的 request_id 不应相同，避免跨会话误去重。"""
        from app.agent.policy_engine import IdempotencyRule

        a = IdempotencyRule.make_request_id("submit_feedback", {"content": "x"}, "sess_1")
        b = IdempotencyRule.make_request_id("submit_feedback", {"content": "x"}, "sess_2")
        assert a != b

    def test_read_tool_not_injected(self):
        """读工具 Schema 未声明 request_id，不应被注入（否则参数校验失败）。"""
        import asyncio
        from app.agent.policy_engine import IdempotencyRule

        rule = IdempotencyRule()
        args = {"query": "兵马俑"}
        asyncio.run(rule.evaluate("search_knowledge", args, {"session_id": "sess_abc"}))
        assert "request_id" not in args

    def test_duplicate_read_is_denied(self):
        """读操作重复调用应被拒绝（省资源）。"""
        import asyncio
        from app.agent.policy_engine import IdempotencyRule

        rule = IdempotencyRule()
        args = {"query": "重复查询"}
        ctx = {"session_id": "sess_dup"}
        first = asyncio.run(rule.evaluate("search_knowledge", args, ctx))
        assert first.allowed
        # 直接模拟 Redis 命中
        with patch.object(IdempotencyRule, "_check_redis", return_value=True):
            second = asyncio.run(rule.evaluate("search_knowledge", args, ctx))
        assert not second.allowed

    def test_duplicate_write_is_allowed_as_retry(self):
        """写操作重复不应在网关拒绝，而应放行交给下游按 request_id 去重。"""
        import asyncio
        from app.agent.policy_engine import IdempotencyRule

        rule = IdempotencyRule()
        args = {"feedback_type": "bug", "content": "重复提交"}
        ctx = {"session_id": "sess_dup_w"}
        with patch.object(IdempotencyRule, "_check_redis", return_value=True):
            decision = asyncio.run(rule.evaluate("submit_feedback", args, ctx))
        assert decision.allowed
        assert decision.metadata.get("is_retry") is True


class TestTimeoutTiering:
    """第三层安全治理：超时分级（读可重试 / 写标 UNKNOWN 禁重试）。"""

    @staticmethod
    def _build_node(fake_tool):
        from app.agent.parallel_tools import ParallelToolNode

        # ToolNode 在构造时根据 tools 建立 tools_by_name 索引，
        # 因此直接用目标工具构造即可（tools_by_name 为只读属性）。
        return ParallelToolNode(tools=[fake_tool])

    def test_write_timeout_marks_unknown_and_not_retryable(self):
        """写操作超时必须标记 UNKNOWN 且不可重试，绝不能提示"失败可重试"。"""
        import asyncio
        from langchain_core.tools import tool

        @tool
        def submit_feedback(feedback_type: str, content: str, shop_name: str = "",
                            request_id: str = "") -> str:
            """写工具。"""
            return "ok"

        async def slow(*a, **kw):
            await asyncio.sleep(5)

        node = self._build_node(submit_feedback)
        node._run_tool_sync = slow

        with patch("app.config.settings.write_tool_timeout_seconds", 0.05):
            msg = asyncio.run(node._execute_single_tool(
                {"name": "submit_feedback", "args": {"feedback_type": "bug", "content": "x"},
                 "id": "call_w"},
                None, {},
            ))

        payload = json.loads(msg.content)
        assert payload["status"] == "UNKNOWN"
        assert payload["retryable"] is False
        assert "禁止重试" in payload["instruction"]
        assert "失败可重试" not in payload["instruction"]

    def test_read_timeout_retries_then_degrades(self):
        """读操作超时应快速重试，重试耗尽后降级返回而非无限重试。"""
        import asyncio
        from langchain_core.tools import tool

        @tool
        def search_knowledge(query: str) -> str:
            """读工具。"""
            return "ok"

        calls = {"n": 0}

        async def slow(*a, **kw):
            calls["n"] += 1
            await asyncio.sleep(5)

        node = self._build_node(search_knowledge)
        node._run_tool_sync = slow

        state = {}
        with patch("app.config.settings.read_tool_timeout_seconds", 0.05), \
             patch("app.config.settings.max_tool_retry_per_call", 1):
            msg = asyncio.run(node._execute_single_tool(
                {"name": "search_knowledge", "args": {"query": "x"}, "id": "call_r"},
                None, state,
            ))

        payload = json.loads(msg.content)
        assert payload["status"] == "TIMEOUT"
        assert payload["retryable"] is False
        assert calls["n"] == 2, "1 次原始调用 + 1 次重试"
        assert state.get("tool_retry_count", {}).get("search_knowledge") == 1

    def test_write_error_marks_unknown(self):
        """写操作异常也可能是"已执行但响应丢失"，同样标记 UNKNOWN 禁重试。"""
        import asyncio
        from langchain_core.tools import tool

        @tool
        def submit_feedback(feedback_type: str, content: str, shop_name: str = "",
                            request_id: str = "") -> str:
            """写工具。"""
            return "ok"

        async def boom(*a, **kw):
            raise ConnectionError("连接被重置")

        node = self._build_node(submit_feedback)
        node._run_tool_sync = boom

        msg = asyncio.run(node._execute_single_tool(
            {"name": "submit_feedback", "args": {"feedback_type": "bug", "content": "x"},
             "id": "call_we"},
            None, {},
        ))
        payload = json.loads(msg.content)
        assert payload["status"] == "UNKNOWN"
        assert payload["retryable"] is False


class TestTaskDeadline:
    """第四层循环控制：任务级总超时（wall-clock）强制终止。"""

    def test_deadline_exceeded_rejects_tool_calls(self):
        """任务总超时后，工具节点应拒绝执行并返回降级指令。"""
        import asyncio
        import time as _time
        from langchain_core.messages import AIMessage
        from app.agent.parallel_tools import ParallelToolNode
        from langchain_core.tools import tool

        @tool
        def search_knowledge(query: str) -> str:
            """读工具。"""
            return "ok"

        node = ParallelToolNode(tools=[search_knowledge])
        state = {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "search_knowledge", "args": {"query": "x"},
                "id": "call_1", "type": "tool_call",
            }])],
            "task_deadline": _time.time() - 1,
        }
        result = asyncio.run(node.run(state))

        payload = json.loads(result["messages"][0].content)
        assert payload["status"] == "DEADLINE_EXCEEDED"
        assert payload["retryable"] is False
        assert "不要再调用任何工具" in payload["instruction"]


class TestAuditTrail:
    """第三层安全治理：审计证据链（谁调用/调什么/参数/决策依据/执行结果）。"""

    def test_audit_record_created_and_backfilled(self):
        """审计记录应包含决策依据，并能回填执行结果。"""
        import asyncio
        from app.agent.policy_engine import PolicyEngine, ToolWhitelistRule, PermissionTierRule

        engine = PolicyEngine()
        engine.add_rule(ToolWhitelistRule())
        engine.add_rule(PermissionTierRule())

        ctx = {"session_id": "sess_audit", "trace_id": "trace_1", "user_id": "u1"}
        decision = asyncio.run(engine.evaluate("search_knowledge", {"query": "兵马俑"}, ctx))
        assert decision.allowed

        trail = engine.get_audit_trail(session_id="sess_audit")
        assert len(trail) == 1
        record = trail[0]
        assert record["tool"] == "search_knowledge"
        assert record["actor"] == "u1"
        assert record["args"] == {"query": "兵马俑"}
        assert record["decision"] == "allow"
        assert record["status"] == "pending"

        ok = engine.record_execution_result(
            "search_knowledge", status="success", result="命中3条", duration_ms=120.0
        )
        assert ok is True
        assert engine.get_audit_trail(session_id="sess_audit")[0]["status"] == "success"

    def test_denied_call_is_audited(self):
        """被拒绝的调用也必须留痕，形成可追溯证据链。"""
        import asyncio
        from app.agent.policy_engine import PolicyEngine, ToolWhitelistRule

        engine = PolicyEngine()
        engine.add_rule(ToolWhitelistRule())

        asyncio.run(engine.evaluate("unknown_tool", {}, {"session_id": "sess_d"}))
        trail = engine.get_audit_trail(session_id="sess_d")
        assert trail[0]["decision"] == "deny"
        assert trail[0]["rule"] == "tool_whitelist"

    def test_audit_summary(self):
        """审计摘要应统计工具分布与决策分布。"""
        import asyncio
        from app.agent.policy_engine import PolicyEngine, ToolWhitelistRule

        engine = PolicyEngine()
        engine.add_rule(ToolWhitelistRule())
        asyncio.run(engine.evaluate("search_knowledge", {}, {"session_id": "s"}))
        asyncio.run(engine.evaluate("bad_tool", {}, {"session_id": "s"}))

        summary = engine.get_audit_summary()
        assert summary["total_records"] == 2
        assert summary["decision_counts"]["allow"] == 1
        assert summary["decision_counts"]["deny"] == 1


class TestMetricsCollector:
    """第五层可观测：延迟 P95 / 失败率 / 重试次数 / 误触发率 / 漏调用率。"""

    def setup_method(self):
        from app.agent.tracer import get_metrics_collector
        get_metrics_collector().reset()

    def test_latency_p95(self):
        """P95 应由真实耗时样本计算。"""
        from app.agent.tracer import get_metrics_collector

        m = get_metrics_collector()
        for i in range(1, 101):
            m.record_tool_call("search_knowledge", duration_ms=float(i), success=True)
        metrics = m.get_metrics()
        assert metrics["tool_calls_total"] == 100
        assert 94 <= metrics["latency_p95_ms"] <= 96

    def test_failure_rate_and_retry_count(self):
        """失败率与重试次数应由真实埋点累计。"""
        from app.agent.tracer import get_metrics_collector

        m = get_metrics_collector()
        m.record_tool_call("t", 10.0, success=True, retries=1)
        m.record_tool_call("t", 20.0, success=False, retries=2, status="failed")
        metrics = m.get_metrics()
        assert metrics["tool_failure_rate"] == "50.00%"
        assert metrics["tool_retry_total"] == 3

    def test_miss_rate(self):
        """漏调用率 = 检出应调未调的次数 ÷ 参与审计的查询数。"""
        from app.agent.tracer import get_metrics_collector

        m = get_metrics_collector()
        m.record_audit(has_miss=True)
        m.record_audit(has_miss=False)
        m.record_audit(has_miss=False)
        m.record_audit(has_miss=False)
        metrics = m.get_metrics()
        assert metrics["audit_queries_total"] == 4
        assert metrics["miss_rate"] == "25.00%"

    def test_false_trigger_rate(self):
        """误触发率 = 空/无效结果次数 ÷ 工具调用总次数。"""
        from app.agent.tracer import get_metrics_collector

        m = get_metrics_collector()
        # 空结果是某次调用产生的结果，因此总调用数需包含它
        for _ in range(5):
            m.record_tool_call("t", 10.0, success=True)
        m.record_empty_result("t")
        metrics = m.get_metrics()
        assert metrics["false_trigger_rate"] == "20.00%"
        assert metrics["call_precision"] == "80.00%"

    def test_write_unknown_counted(self):
        """写操作 UNKNOWN 次数应单独统计（风险监控）。"""
        from app.agent.tracer import get_metrics_collector

        m = get_metrics_collector()
        m.record_tool_call("submit_feedback", 10.0, success=False,
                           is_write=True, status="unknown")
        metrics = m.get_metrics()
        assert metrics["write_calls_total"] == 1
        assert metrics["write_unknown_total"] == 1

    def test_empty_metrics_no_division_error(self):
        """无数据时不应抛除零异常。"""
        from app.agent.tracer import get_metrics_collector

        metrics = get_metrics_collector().get_metrics()
        assert metrics["tool_failure_rate"] == "0%"
        assert metrics["miss_rate"] == "N/A"
        assert metrics["latency_p95_ms"] == 0.0


class TestConfirmExecution:
    """第三层安全治理：人工确认闭环必须真正执行写操作（不能停在等待确认）。"""

    def setup_method(self):
        from app.agent.policy_engine import WriteConfirmRule
        WriteConfirmRule._pending_confirms.clear()

    def test_confirm_token_is_one_time(self):
        """确认令牌一次性使用，取出后即失效。"""
        import asyncio
        from app.agent.policy_engine import WriteConfirmRule

        rule = WriteConfirmRule()
        decision = asyncio.run(rule.evaluate(
            "submit_feedback",
            {"feedback_type": "bug", "content": "页面打不开", "request_id": "req_abc"},
            {"session_id": "sess_c1"},
        ))
        token = decision.confirm_token

        pending = WriteConfirmRule._pending_confirms.pop(token)
        assert pending["tool_name"] == "submit_feedback"
        assert pending["args"]["request_id"] == "req_abc"
        assert token not in WriteConfirmRule._pending_confirms

    def test_confirm_rejects_cross_session(self):
        """确认令牌必须校验会话归属，防止跨会话盗用。"""
        import asyncio
        from app.agent.policy_engine import WriteConfirmRule
        from app.routers.chat import confirm_action

        rule = WriteConfirmRule()
        decision = asyncio.run(rule.evaluate(
            "submit_feedback",
            {"feedback_type": "bug", "content": "x"},
            {"session_id": "sess_owner"},
        ))

        result = asyncio.run(confirm_action(decision.confirm_token, "sess_attacker"))
        assert result["success"] is False
        assert "不匹配" in result["message"]

    def test_confirm_executes_write_tool(self):
        """确认后必须真正调用写工具并返回执行结果。"""
        import asyncio
        from app.agent.policy_engine import WriteConfirmRule
        from app.routers.chat import confirm_action

        rule = WriteConfirmRule()
        decision = asyncio.run(rule.evaluate(
            "submit_feedback",
            {"feedback_type": "bug", "content": "页面打不开", "request_id": "req_xyz"},
            {"session_id": "sess_exec"},
        ))

        with patch("app.agent.tools.ALL_TOOLS") as fake_tools:
            fake = Mock()
            fake.name = "submit_feedback"
            fake.invoke.return_value = '{"success": true, "feedback_id": "fbk_1"}'
            fake_tools.__iter__ = Mock(return_value=iter([fake]))

            result = asyncio.run(confirm_action(decision.confirm_token, "sess_exec"))

        assert result["success"] is True
        assert result["tool_name"] == "submit_feedback"
        assert result["request_id"] == "req_xyz"
        assert "fbk_1" in result["result"]
        fake.invoke.assert_called_once()

    def test_confirm_invalid_token(self):
        """无效令牌应被拒绝。"""
        import asyncio
        from app.routers.chat import confirm_action

        result = asyncio.run(confirm_action("confirm_not_exist", "sess_x"))
        assert result["success"] is False


# ==================== 运行入口 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("🛡️  防御体系V3.0 测试套件")
    print("=" * 70)

    # 运行pytest
    pytest.main([__file__, "-v", "--tb=short"])