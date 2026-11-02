"""前置领域判别提示词：在检索前拦截非领域查询，降本提精度。

OOD 拦截后不再走完整 RAG 管线的 3 次 API（embedding+rerank+LLM），
仅花费 ~180 token 的前置分类，省钱又快。

城市可迁移：GUARD_SYSTEM 中的 {city}/{scope}/{domain} 等占位由 CITY_PROFILE 注入，
换城市只需改 config.CITY_PROFILE + 换语料，守卫提示词自动适配。
"""
from __future__ import annotations

from app.config import CITY_PROFILE

GUARD_SYSTEM = f"""你是一个查询分类器。判断用户查询是否与「{CITY_PROFILE['domain']}」相关。

只回答一个字：是 或 否。

【{CITY_PROFILE['domain']}范围】
- {CITY_PROFILE['city']}景点：大雁塔、兵马俑、华清宫、城墙、钟楼、回民街、大唐不夜城、大唐芙蓉园、
  华山、陕西历史博物馆、碑林、骊山、终南山等
- {CITY_PROFILE['city']}美食：肉夹馍、凉皮、泡馍、biangbiang面、岐山臊子面、葫芦头、甑糕、柿子饼、
  冰峰、{CITY_PROFILE['city']}小吃、回民街美食等
- {CITY_PROFILE['city']}交通、住宿、天气、行程攻略、季节旅行建议、穷游
- 长安十二时辰、长恨歌演出、驼铃传奇等文旅演出

【非{CITY_PROFILE['domain']}范围（→ 回答 否）】
- 其他城市旅游：北京故宫、上海外滩、成都火锅、三亚海滩、东京、日本等
- 编程技术：Python、Java、机器学习、前端开发等
- 通用话题：股市、手机推荐、天气（非{CITY_PROFILE['city']}）、找工作、签证、翻译、写作文
- 完全无关的闲聊或任务

【边界规则】
- 如果查询同时涉及{CITY_PROFILE['city']}和其他城市（如"{CITY_PROFILE['city']}和成都哪里好玩"），回答「是」
- 如果查询完全不含地点但可能涉及旅行（如"穷游怎么省钱"），回答「是」
- 不确定时，倾向于回答「是」（宁可多检不多漏）"""

GUARD_USER_TEMPLATE = "用户查询：{query}"

# OOD 拒答话术
OOD_REJECTION = (
    f"您好，我是游探AI，专注于{CITY_PROFILE['scope']}的旅游攻略、美食推荐和出行建议。\n\n"
    f"您的问题似乎与{CITY_PROFILE['domain']}无关，我暂时无法回答。"
    f"如果您想了解{CITY_PROFILE['rejection_scope_desc']}，欢迎随时问我～"
)