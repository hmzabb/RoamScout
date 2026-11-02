"""Agent 状态定义。

- MessagesState 内置 messages 字段 + add_messages reducer——每次节点返回的新消息
  【追加】到历史而不是覆盖，这正是多轮工具调用的记忆基础；
- 自定义 iter 字段记录工具调用轮数，配合 recursion_limit 双重防死循环；
- trace_id 字段贯穿全链路，用于第五层可观测；
- session_id 用于预算跟踪和策略引擎；
- round_input_tokens / round_output_tokens 用于第四层 Token 预算；
- param_retry / need_param_retry 用于第二层参数校验失败后的"回传错误让模型修正"闭环；
- task_deadline 是第四层的任务级总超时（wall-clock），超时后强制终止；
- tool_retry_count 记录每个工具的已重试次数，配合 max_tool_retry_per_call 做失败退出。
"""
from typing import Dict, Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Agent 状态：messages（累加）+ iter（本轮已执行的工具调用轮数）+ trace 元信息。"""

    iter: int = 0
    trace_id: Optional[str] = None
    session_id: str = ""
    round_input_tokens: int = 0
    round_output_tokens: int = 0
    param_retry: int = 0
    need_param_retry: bool = False
    task_deadline: Optional[float] = None
    tool_retry_count: Dict[str, int] = {}