"""景点百科服务：高德POI + Milvus知识库双重数据源。

数据策略：
1. 高德 POI API → 实时结构化数据（地址/评分/营业时间等）
2. Milvus 向量库 → 历史文化/攻略/深度内容
3. 两路结果融合，返回完整百科信息

API 参考：
- 高德 POI 搜索：https://restapi.amap.com/v3/place/text
- 高德 POI 详情：https://restapi.amap.com/v3/place/detail

缓存策略：景点百科 TTL=3600秒（1小时），结构性数据变化频率低。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 高德地图 API 基地址
AMAP_API_BASE = "https://restapi.amap.com/v3"

# 西安著名景点预设信息（离线兜底，API不可用时使用）
XI_AN_ATTRACTIONS_PRESET: Dict[str, Dict[str, Any]] = {
    "兵马俑": {
        "name": "秦始皇兵马俑博物馆",
        "address": "西安市临潼区秦陵北路",
        "ticket": "120元（旺季）/ 100元（淡季）",
        "open_time": "08:30-17:00（16:30停止检票）",
        "rating": "4.8",
        "summary": "世界第八大奇迹，出土陶俑8000余件，展现秦代军事阵容",
        "tips": "建议游览3-4小时，避开节假日高峰",
        "transport": "地铁9号线秦岭西站，换乘游5/306路公交",
    },
    "大雁塔": {
        "name": "大雁塔·大慈恩寺",
        "address": "西安市雁塔区雁塔南路",
        "ticket": "大慈恩寺50元，登塔另收30元",
        "open_time": "08:00-18:00（夏季17:30停售）",
        "rating": "4.7",
        "summary": "唐代玄奘法师为保存经卷而建，西安地标，世界文化遗产",
        "tips": "北广场音乐喷泉每晚演出，建议傍晚前往",
        "transport": "地铁3/4号线大雁塔站",
    },
    "城墙": {
        "name": "西安城墙",
        "address": "西安市中心，环绕古城一周",
        "ticket": "54元",
        "open_time": "08:00-21:00（南门延长至22:00）",
        "rating": "4.7",
        "summary": "中国现存规模最大、保存最完整的古代城垣，全长13.74公里",
        "tips": "建议租自行车游览（约2小时环城），南门夜景最佳",
        "transport": "地铁2号线永宁门站（南门）",
    },
    "华清宫": {
        "name": "华清宫（华清池）",
        "address": "西安市临潼区华清路38号",
        "ticket": "120元（旺季）/ 100元（淡季）",
        "open_time": "07:00-18:00",
        "rating": "4.6",
        "summary": "唐代皇家温泉行宫，唐玄宗与杨贵妃的爱情故事发生地",
        "tips": "可结合兵马俑一同游览（相距约15分钟车程）",
        "transport": "地铁9号线华清池站",
    },
    "钟楼": {
        "name": "西安钟楼",
        "address": "西安市碑林区东大街与西大街交汇处",
        "ticket": "35元（钟鼓楼联票50元）",
        "open_time": "08:30-21:00",
        "rating": "4.6",
        "summary": "中国现存钟楼中规模最大、保存最完整的一座，始建于明洪武年间",
        "tips": "紧邻回民街，可一并游览品尝美食",
        "transport": "地铁2号线钟楼站",
    },
    "回民街": {
        "name": "回民街（北院门）",
        "address": "西安市莲湖区北院门",
        "ticket": "免费",
        "open_time": "全天（店铺通常10:00-22:00）",
        "rating": "4.5",
        "summary": "西安最著名的小吃街，汇聚数百种特色美食和手工艺品",
        "tips": "推荐傍晚前往，热闹且美食种类最全",
        "transport": "地铁2号线钟楼站，步行约5分钟",
    },
    "大唐不夜城": {
        "name": "大唐不夜城",
        "address": "西安市雁塔区雁塔南路",
        "ticket": "免费",
        "open_time": "全天（夜景最佳，18:00亮灯）",
        "rating": "4.7",
        "summary": "以盛唐文化为主题的步行街，灯光璀璨，网红不倒翁小姐姐所在地",
        "tips": "建议晚上去，灯光秀和表演从19:00开始",
        "transport": "地铁3/4号线大雁塔站，步行约5分钟",
    },
    "华山": {
        "name": "华山风景名胜区",
        "address": "渭南市华阴市华山镇",
        "ticket": "180元（旺季）/ 100元（淡季）",
        "open_time": "全天（索道07:00-19:00）",
        "rating": "4.9",
        "summary": "五岳之西岳，以险峻著称，长空栈道和鹞子翻身是著名险道",
        "tips": "建议缆车上下（北峰索道80元），穿登山鞋，带足水",
        "transport": "西安北站高铁至华山北站（约30分钟），转景区大巴",
    },
}


@dataclass
class AttractionInfo:
    """景点信息数据结构。"""

    name: str = ""
    address: str = ""
    ticket: str = ""
    open_time: str = ""
    rating: str = ""
    summary: str = ""
    tips: str = ""
    transport: str = ""
    source: str = ""  # "amap" / "preset" / "milvus"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "ticket": self.ticket,
            "open_time": self.open_time,
            "rating": self.rating,
            "summary": self.summary,
            "tips": self.tips,
            "transport": self.transport,
            "source": self.source,
        }


class AmapClient:
    """高德地图 API 客户端。

    使用方式：
        client = AmapClient(api_key=settings.amap_api_key)
        result = client.search_poi("大雁塔", "西安")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "amap_api_key", "")
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=10.0,
                verify=False,
                headers={"User-Agent": "YouTanAI/1.0"},
            )
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # POI 搜索（景点）
    # ------------------------------------------------------------------

    def search_attraction(
        self, name: str, city: str = "西安"
    ) -> Optional[Dict[str, Any]]:
        """搜索景点信息。

        Args:
            name: 景点名称（如"大雁塔""兵马俑"）
            city: 城市名，默认西安

        Returns:
            景点信息字典，未找到返回 None
        """
        try:
            resp = self.client.get(
                f"{AMAP_API_BASE}/place/text",
                params={
                    "keywords": name,
                    "city": city,
                    "citylimit": "true",
                    "types": "风景名胜|公园广场|纪念馆|寺庙道观|国家级景点|城市广场|博物馆|展览馆|文化场馆",
                    "key": self.api_key,
                    "offset": 10,
                    "extensions": "all",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                info = data.get("info", "")
                if "CUQPS" in info or "OVER_LIMIT" in info:
                    logger.warning(f"[Amap] 景点POI搜索触发QPS限流: {info}")
                else:
                    logger.warning(f"[Amap] 景点POI搜索接口异常: {info}")
                return None
            if not data.get("pois"):
                logger.warning(f"[Amap] 未找到景点「{name}」")
                return None

            best = self._pick_best_poi(name, data["pois"])
            if best is None:
                return None

            biz = best.get("biz_ext") or {}
            return {
                "name": best.get("name", name),
                "address": best.get("address", ""),
                "rating": biz.get("rating", "") or "",
                "type": best.get("type", ""),
                "location": best.get("location", ""),  # "lng,lat"
                "photos": [
                    p.get("url", "") for p in (best.get("photos", []) or [])[:3]
                ],
                "pcode": best.get("pcode", ""),
            }
        except Exception as e:
            logger.error(f"[Amap] POI搜索异常: {e}")
            return None

    # 附属设施关键词：这些 POI 不是景点本体，排序时降权
    _ACCESSORY_KEYWORDS = (
        "乘车点", "直通车", "售票", "入口", "出口", "停车场", "检票",
        "游客中心", "派出所", "管理处", "洗手间", "卫生间", "商店",
        "便利店", "公交站", "文创", "纪念品", "超市", "商业街", "连接线",
        "雕像", "灯笼", "古玩城", "旗舰店", "专营店", "专卖店", "购物",
    )

    # 商业/附属 POI 的 type 特征，命中则直接排除
    _COMMERCIAL_TYPE_KEYWORDS = (
        "购物服务", "专卖店", "专营店", "特色商业街", "体育休闲服务",
        "休闲场所", "展览馆",
    )

    def _type_tier(self, poi: Dict[str, Any]) -> int:
        """按 POI 的 type 字段评估「本体性」，值越小越像景点本体。

        - 0：国家级景点（高德认定的核心景区）
        - 1：景区/文博本体（风景名胜;风景名胜;xxx / 公园广场 / 寺庙道观 / 博物馆 / 纪念馆）
        - 2：风景名胜相关（附属游览点）
        - 3：商业或非景区类（购物、休闲、展览等），直接排除
        """
        poi_type = poi.get("type", "") or ""
        if any(k in poi_type for k in self._COMMERCIAL_TYPE_KEYWORDS):
            return 3
        if "国家级景点" in poi_type:
            return 0
        if "风景名胜相关" in poi_type:
            return 2
        if any(
            k in poi_type
            for k in (
                "风景名胜;风景名胜;", "公园广场", "寺庙道观",
                "博物馆", "纪念馆", "文化场馆",
            )
        ):
            return 1
        return 3

    def _pick_best_poi(self, name: str, pois: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从候选 POI 中选出最匹配景点本体的结果。

        排序键为 (type 本体性, 名称匹配度, -评分)：
        - type 本体性优先，保证「秦始皇帝陵博物院」胜过「兵马俑文创店」；
        - 名称匹配度：完全相等 < 以查询词开头 < 包含查询词 < 不包含；
        - 最后以评分、是否有坐标作为补充。

        接受条件：POI 必须是国家级景点，或名称与查询词存在包含关系；
        否则视为未匹配（返回 None），交由预设数据/知识库兜底，
        避免把「回民街旗舰店」这类附属 POI 误当作景点本体。

        Args:
            name: 用户查询的景点名
            pois: 高德返回的 POI 列表

        Returns:
            最匹配的 POI，无合适结果时返回 None
        """
        if not pois:
            return None

        def name_tier(poi: Dict[str, Any]) -> int:
            poi_name = poi.get("name", "")
            accessory = any(k in poi_name for k in self._ACCESSORY_KEYWORDS)
            if poi_name == name:
                return 0
            if name in poi_name:
                return 2 if accessory else 1
            return 3

        def sort_key(poi: Dict[str, Any]) -> tuple:
            biz = poi.get("biz_ext") or {}
            try:
                rating = float(biz.get("rating") or 0)
            except (ValueError, TypeError):
                rating = 0.0
            has_loc = 1 if poi.get("location") else 0
            return (self._type_tier(poi), name_tier(poi), -rating, -has_loc)

        candidates = [
            p for p in pois
            if self._type_tier(p) < 3
            and (self._type_tier(p) == 0 or name_tier(p) <= 2)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=sort_key)[0]

    # ------------------------------------------------------------------
    # 路径规划（公交/驾车/步行）
    # ------------------------------------------------------------------

    # 路径规划地名别名：口语名称 → 高德可精确命中的规范名称
    # 依据：实测「兵马俑」在高德中无同名 POI，其本体登记名为「秦始皇帝陵博物院」
    _RESOLVE_ALIASES: Dict[str, str] = {
        "兵马俑": "秦始皇帝陵博物院",
        "华清池": "华清宫",
        "秦始皇兵马俑": "秦始皇帝陵博物院",
    }

    # 路径端点应避开的附属设施（含这些词的 POI 不是地点本体）
    _RESOLVE_ACCESSORY = (
        "乘车点", "直通车", "售票", "停车场", "入口", "出口", "检票",
        "游客中心", "洗手间", "卫生间", "商店", "超市", "文创",
        "纪念品", "商业街", "连接线", "派出所", "管理处",
    )

    def _resolve_location(self, name: str, city: str) -> Optional[str]:
        """把地名解析为高德所需的 "lng,lat" 坐标。

        高德路径规划接口只接受经纬度，不接受地名，因此需先做 POI 搜索。
        若传入的已是 "lng,lat" 格式则直接返回。

        为避免「火星基地」这类不存在的地名被模糊匹配成「星火团建露营基地」，
        要求候选 POI 名称必须与查询词存在包含关系，否则视为无法解析。

        Args:
            name: 地名或坐标
            city: 城市名

        Returns:
            "lng,lat" 坐标字符串，解析失败返回 None
        """
        if not name:
            return None

        parts = name.split(",")
        if len(parts) == 2:
            try:
                float(parts[0].strip())
                float(parts[1].strip())
                return name
            except ValueError:
                pass

        keyword = self._RESOLVE_ALIASES.get(name.strip(), name.strip())

        try:
            resp = self.client.get(
                f"{AMAP_API_BASE}/place/text",
                params={
                    "keywords": keyword,
                    "city": city,
                    "citylimit": "true",
                    "key": self.api_key,
                    "offset": 10,
                },
            )
            data = resp.json()
            if data.get("status") != "1":
                info = data.get("info", "")
                if "CUQPS" in info or "OVER_LIMIT" in info:
                    logger.warning(f"[Amap] 地名解析触发QPS限流: {info}")
                else:
                    logger.warning(f"[Amap] 地名解析接口异常: {info}")
                return None
            pois = data.get("pois") or []

            def is_relevant(poi: Dict[str, Any]) -> bool:
                poi_name = poi.get("name", "")
                return keyword in poi_name or poi_name in keyword

            def is_accessory(poi: Dict[str, Any]) -> bool:
                poi_name = poi.get("name", "")
                return any(k in poi_name for k in self._RESOLVE_ACCESSORY)

            candidates = [
                p for p in pois
                if p.get("location") and is_relevant(p) and not is_accessory(p)
            ]
            if not candidates:
                logger.warning(
                    f"[Amap] 地名「{name}」未解析到坐标（无同名POI，"
                    f"返回{len(pois)}条近似结果），放弃解析"
                )
                return None

            candidates.sort(key=lambda p: 0 if p.get("name") == keyword else 1)
            return candidates[0].get("location")
        except Exception as e:
            logger.error(f"[Amap] 地名转坐标异常: {e}")
        return None

    def get_transit_direction(
        self,
        origin: str,
        destination: str,
        city: str = "西安",
        mode: str = "transit",
    ) -> Optional[Dict[str, Any]]:
        """获取出行路径规划。

        Args:
            origin: 出发地（地名或坐标 "lng,lat"）
            destination: 目的地（地名或坐标 "lng,lat"）
            city: 城市
            mode: 出行方式
                - "transit": 公交/地铁（默认）
                - "driving": 驾车
                - "walking": 步行
                - "bicycling": 骑行

        Returns:
            路径规划结果字典，失败返回 None
        """
        mode_map = {
            "transit": "/direction/transit/integrated",
            "driving": "/direction/driving",
            "walking": "/direction/walking",
            "bicycling": "/direction/bicycling",
        }
        endpoint = mode_map.get(mode, mode_map["transit"])

        origin_coord = self._resolve_location(origin, city)
        dest_coord = self._resolve_location(destination, city)
        if not origin_coord or not dest_coord:
            logger.warning(
                f"[Amap] 无法解析起点/终点坐标: origin={origin}->{origin_coord}, "
                f"destination={destination}->{dest_coord}"
            )
            return None

        params: Dict[str, Any] = {
            "origin": origin_coord,
            "destination": dest_coord,
            "key": self.api_key,
            "extensions": "all",
        }
        if mode == "transit":
            params["city"] = city
            params["cityd"] = city

        try:
            resp = self.client.get(
                f"{AMAP_API_BASE}{endpoint}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                logger.warning(f"[Amap] 路径规划失败: {data.get('info', '')} | {data.get('infocode', '')}")
                return None

            route = data.get("route", {})
            paths = route.get("transits") or route.get("paths") or []
            if not paths:
                return None

            best = paths[0]
            result: Dict[str, Any] = {
                "origin": origin,
                "destination": destination,
                "mode": mode,
            }

            if mode == "transit":
                result["duration"] = best.get("duration", "")
                result["distance"] = best.get("distance", "")
                result["cost"] = best.get("cost", "未知")
                result["walking_distance"] = best.get("walking_distance", "")
                segments = best.get("segments", [])
                result["steps"] = []
                for seg in segments[:5]:
                    walking = seg.get("walking", {}) or {}
                    bus = seg.get("bus", {}) or {}
                    railway = seg.get("railway", {}) or {}
                    buslines = bus.get("buslines") or []

                    if buslines:
                        bl = buslines[0]
                        result["steps"].append({
                            "type": "bus",
                            "name": bl.get("name", ""),
                            "departure": (bl.get("departure_stop") or {}).get("name", ""),
                            "arrival": (bl.get("arrival_stop") or {}).get("name", ""),
                            "duration": seg.get("duration", ""),
                        })
                    elif railway.get("name"):
                        result["steps"].append({
                            "type": "railway",
                            "name": railway.get("name", ""),
                            "departure": (railway.get("departure_stop") or {}).get("name", ""),
                            "arrival": (railway.get("arrival_stop") or {}).get("name", ""),
                            "duration": seg.get("duration", ""),
                        })
                    elif walking:
                        result["steps"].append({
                            "type": "walking",
                            "distance": walking.get("distance", ""),
                            "duration": walking.get("duration", ""),
                        })
            else:
                result["duration"] = best.get("duration", "")
                result["distance"] = best.get("distance", "")
                steps = best.get("steps", [])
                result["steps_count"] = len(steps)

            return result
        except Exception as e:
            logger.error(f"[Amap] 路径规划异常: {e}")
            return None


# ==================== 简单内存缓存 ====================

@dataclass
class _CacheEntry:
    data: Any
    expire_at: float


class AttractionCache:
    """景点数据缓存（TTL 1小时）。"""

    def __init__(self, ttl: float = 3600.0):
        self.ttl = ttl
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and time.time() < entry.expire_at:
            return entry.data
        if entry:
            del self._store[key]
        return None

    def set(self, key: str, data: Any):
        self._store[key] = _CacheEntry(data=data, expire_at=time.time() + self.ttl)


_attraction_cache = AttractionCache(ttl=3600.0)


def get_amap_client() -> AmapClient:
    return AmapClient()


def _fuzzy_match_preset(name: str) -> Optional[str]:
    """模糊匹配预设景点名。"""
    for key in XI_AN_ATTRACTIONS_PRESET:
        if key in name or name in key:
            return key
    # 更宽松的匹配：要求重合字符占比足够高，避免「陕西历史博物馆」误配到「城墙」
    name_chars = set(name)
    for key in XI_AN_ATTRACTIONS_PRESET:
        key_chars = set(key)
        common = len(name_chars & key_chars)
        shorter = min(len(name_chars), len(key_chars))
        if common >= 2 and shorter and common / shorter >= 0.6:
            return key
    return None


def get_attraction_info(name: str) -> Optional[Dict[str, Any]]:
    """获取景点百科信息（多源融合）。

    优先级：高德POI > 预设数据 > Milvus知识库

    Args:
        name: 景点名称

    Returns:
        景点信息字典，未找到返回 None
    """
    import json as _json

    cache_key = f"attraction:{name}"
    cached = _attraction_cache.get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Any] = {"name": name}

    # 1. 尝试高德 POI
    client = get_amap_client()
    amap_data = None
    try:
        amap_data = client.search_attraction(name, "西安")
        if amap_data:
            result["name"] = amap_data.get("name", name)
            result["address"] = amap_data.get("address", "")
            result["rating"] = amap_data.get("rating", "")
            result["source"] = "amap"
            # 从坐标反推大致位置描述
            loc = amap_data.get("location", "")
            if loc:
                result["location"] = loc
    except Exception as e:
        logger.warning(f"[Attraction] 高德查询失败，使用预设兜底: {e}")
    finally:
        client.close()

    # 2. 用预设数据补齐门票/时间/攻略等信息
    matched_preset = _fuzzy_match_preset(name)
    if matched_preset:
        preset = XI_AN_ATTRACTIONS_PRESET[matched_preset]
        for field in ("ticket", "open_time", "summary", "tips", "transport"):
            if field not in result or not result.get(field):
                result[field] = preset.get(field, "")
        # 预设数据的景点名/地址更规范（高德可能返回「大雁塔北广场」这类子 POI），
        # 因此命中预设时以预设为准；高德原始名称仅记录日志，不进入 Agent 上下文
        if preset.get("name"):
            if amap_data and result.get("name") != preset["name"]:
                logger.debug(
                    f"[Attraction] 「{name}」高德POI名称={result.get('name')}，"
                    f"以预设名称={preset['name']} 为准"
                )
            result["name"] = preset["name"]
        if preset.get("address"):
            result["address"] = preset["address"]
        if not result.get("source"):
            result["source"] = "preset"

    # 3. 尝试 Milvus 知识库补充历史文化内容
    try:
        from app.repositories.vector_store import get_store
        from app.services.embedding import get_embedding_client

        qv = get_embedding_client().embed_texts([f"{name} 景点介绍 历史文化 游览攻略"])
        if qv:
            hits = get_store().query(
                qv[0], top_k=3, where={"source": "corpus"}
            )
            if hits:
                knowledge_texts = [
                    h.get("text", "")[:200]
                    for h in hits
                    if h.get("score", 0) > 0.5
                ]
                if knowledge_texts:
                    result["knowledge"] = knowledge_texts
                    if not result.get("summary"):
                        result["summary"] = knowledge_texts[0][:100]
                    if not result.get("source"):
                        result["source"] = "milvus"
    except Exception as e:
        logger.debug(f"[Attraction] Milvus查询失败（非致命）: {e}")

    if not result.get("source"):
        return None

    _attraction_cache.set(cache_key, result)
    return result


def get_transit_info(
    origin: str,
    destination: str,
    city: str = "西安",
    mode: str = "transit",
) -> Optional[Dict[str, Any]]:
    """获取出行交通信息。

    Args:
        origin: 出发地
        destination: 目的地
        city: 城市
        mode: 出行方式

    Returns:
        路径规划结果字典
    """
    import json as _json

    cache_key = f"transit:{origin}:{destination}:{mode}:{city}"
    cached = _attraction_cache.get(cache_key)
    if cached is not None:
        return cached

    client = get_amap_client()
    try:
        result = client.get_transit_direction(origin, destination, city, mode)
        if result:
            _attraction_cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"[Attraction] 路径规划异常: {e}")
        return None
    finally:
        client.close()


def format_attraction_for_agent(name: str) -> str:
    """格式化景点百科数据为 Agent 可读的 JSON 字符串。

    Args:
        name: 景点名称

    Returns:
        JSON 字符串
    """
    import json as _json

    info = get_attraction_info(name)
    if info is None:
        return _json.dumps({
            "error": f"暂未找到「{name}」的详细信息，建议换个关键词重试（如「大雁塔」「兵马俑」「城墙」等）",
            "suggestion": "可尝试搜索更通用的景点名称",
        }, ensure_ascii=False)

    return _json.dumps(info, ensure_ascii=False)


def format_transit_for_agent(
    origin: str,
    destination: str,
    city: str = "西安",
    mode: str = "transit",
) -> str:
    """格式化交通信息为 Agent 可读的 JSON 字符串。"""
    import json as _json

    info = get_transit_info(origin, destination, city, mode)
    if info is None:
        mode_names = {"transit": "公交/地铁", "driving": "驾车", "walking": "步行", "bicycling": "骑行"}
        mode_name = mode_names.get(mode, mode)
        return _json.dumps({
            "error": f"暂未查到从「{origin}」到「{destination}」的{mode_name}方案",
            "suggestion": "请尝试修改出发地/目的地，或选择其他出行方式",
        }, ensure_ascii=False)

    return _json.dumps(info, ensure_ascii=False)