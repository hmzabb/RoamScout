"""全局配置：pydantic-settings 读取 .env，对应 Java 侧 application.yaml。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# changan_ai 项目根目录（config.py 在 app/ 下，上级即根）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 从 .env 读取模型配置
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 大模型（DeepSeek，OpenAI 兼容协议）----
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---- Embedding（DeepSeek 无 embedding 接口，走硅基流动）----
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"

    # ---- 外部 API（天气 / 景点 / 交通）----
    # 和风天气 API（免费版 1000次/天）：https://dev.qweather.com/
    qweather_api_key: str = ""
    # 和风天气专属 API Host（新版控制台在「项目管理」页展示，如 abc123.re.qweatherapi.com）
    # 留空则使用旧域名 geoapi/devapi.qweather.com
    qweather_api_host: str = ""
    # 高德地图 API（免费版 5000次/天）：https://lbs.amap.com/
    amap_api_key: str = ""

    # ---- 评分模型（RAGAS evaluator，跨家族：Qwen 评 DeepSeek，避免自评偏差）----
    # 阿里云百炼 DashScope（OpenAI 兼容协议）
    qwen_eval_api_key: str = ""
    qwen_eval_base_url: str = "https://ws-5trhkwz426cq0en9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    qwen_eval_model: str = "qwen3.7-plus"

    # ---- 重排（硅基流动 bge-reranker-v2-m3，免费 API）----
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---- Java 后端（数据源，Python 不直连 MySQL）----
    java_base_url: str = "http://127.0.0.1:8081"
    java_timeout_seconds: float = 3.0

    # ---- RAG 参数 ----
    rag_recall_top_k: int = 8
    rag_rerank_top_k: int = 4
    rag_min_score: float = 0.45
    rag_recall_min_score: float = 0.45  # 原始召回最低相似度(P5分位)，过滤明显噪声
    rag_max_context_chars: int = 6000

    # ---- Milvus 向量库（Docker standalone，gRPC 19530 / HTTP 9091）
    # pymilvus 3.x 要求带协议前缀：gRPC 用 tcp://，HTTP 用 http://
    milvus_uri: str = "tcp://127.0.0.1:19530"

    # ---- Redis（Java 端已有 127.0.0.1:6379，复用即可）----
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_password: str = ""
    redis_session_ttl_seconds: int = 1800  # 30分钟自动过期

    # ---- 记忆管理（摘要记忆）----
    summary_enabled: bool = True  # 是否启用摘要记忆
    summary_max_rounds: int = 3  # 超过这个轮数就触发摘要（与MAX_ROUNDS一致）
    summary_model_temperature: float = 0.1  # 摘要用低温度（更确定性）

    # ---- 服务 ----
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # ---- Function Calling 五层防御体系 ----
    # 第二层·防护1：约束解码（Grammar-Constrained Decoding）
    # 推理引擎后端：api（云API+响应侧修复）| vllm（guided_json）| llamacpp（GBNF）
    guided_decoding_backend: str = "api"
    constrained_decoding_enabled: bool = True  # 启用响应侧语法约束
    constrained_decode_max_repair: int = 1  # JSON 语法修复失败后的重试次数
    constrained_decode_retry_on_error: bool = True  # 参数校验失败时回传错误让模型自我修正

    # 第二层·防护2：参数校验（JSON Schema 编译校验）
    param_validation_enabled: bool = True  # 启用执行前 Schema 级参数校验
    param_validation_max_retry: int = 2  # 同一轮内模型修正参数的最大重试次数

    # 第二层·防护3：工具检索（语义 + 关键词混合 Top-K）
    dynamic_tool_retrieval_enabled: bool = True  # 启用后根据用户意图动态筛选工具
    max_tools_per_request: int = 5  # 最终绑定给LLM的工具数上限
    tool_retrieval_top_k: int = 10  # 检索候选池大小（Top-K，工具规模大时调至10~20）
    tool_retrieval_semantic_weight: float = 0.7  # 语义相似度权重
    tool_retrieval_keyword_weight: float = 0.3  # 关键词命中权重
    tool_retrieval_min_score: float = 0.15  # 候选池最低入选分数，低于则丢弃
    tool_retrieval_cache_ttl: int = 86400  # 工具向量索引缓存时间（秒）

    # 第三层：Policy Engine（安全治理层）
    policy_engine_enabled: bool = True  # 启用策略引擎
    policy_write_requires_confirm: bool = True  # 写操作需要人工确认
    policy_money_requires_confirm: bool = True  # 涉及金额的操作需要人工确认
    idempotency_persist: bool = True  # 幂等键持久化到Redis
    idempotency_inject_request_id: bool = True  # 写操作自动注入 request_id 供下游去重

    # 第四层：循环控制（轮数 + 预算 + 总超时）
    token_budget_enabled: bool = True  # 启用token预算
    max_total_tokens_per_session: int = 16000  # 单会话最大token消耗
    max_tokens_per_round: int = 4000  # 单轮最大token消耗
    warning_token_ratio: float = 0.8  # 达到80%预算时告警
    max_task_duration_seconds: float = 90.0  # 单任务总超时（wall-clock），超时强制终止
    read_tool_timeout_seconds: float = 15.0  # 读操作单次超时（可快速重试）
    write_tool_timeout_seconds: float = 30.0  # 写操作单次超时（超时标记UNKNOWN，不可重试）
    max_tool_retry_per_call: int = 2  # 单个工具调用允许的重试次数上限

    # 第五层：Trace（可观测层）
    tracing_enabled: bool = True  # 启用trace
    trace_sample_rate: float = 1.0  # 采样率（1.0=全量，0.1=10%）
    trace_redis_ttl: int = 3600  # trace在Redis中的保留时间（秒）
    metrics_enabled: bool = True  # 启用指标聚合（延迟P95/失败率/重试次数等）

    # 工具Schema版本
    tool_schema_version: str = "1.0.0"


# =============================================================================
# 城市/领域画像：换城市只需改这里 + 换语料，代码零改动
# =============================================================================
CITY_PROFILE = {
    "city": "西安",                     # 城市名
    "scope": "西安及周边",               # 地域范围（RAG/Agent 提示词中拼接）
    "domain": "西安文旅",                # 领域标签（守卫分类器使用）
    "app_name": "游探APP",              # APP名称
    "rejection_scope_desc": "西安的景点、美食或游玩攻略",  # 拒答引导语
}


settings = Settings()