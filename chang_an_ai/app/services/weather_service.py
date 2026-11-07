"""和风天气 API 服务封装。

QWeather API 免费版：1000次/天，足够日常使用。
文档：https://dev.qweather.com/docs/api/

支持接口：
- 城市搜索（LocationID 查询）
- 实时天气
- 7天天气预报
- 生活指数（穿衣/紫外线/洗车等）

数据缓存策略：天气数据 TTL=5分钟，避免频繁请求 API。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 和风天气 API 基地址（兼容新旧两种部署方式）
# 旧域名：geoapi.qweather.com / devapi.qweather.com
# 新域名：控制台为每个项目分配的专属 Host（如 abc123.re.qweatherapi.com）
def _resolve_qweather_bases() -> tuple[List[str], str]:
    """解析和风天气接口基地址。

    新版控制台为每个项目分配专属 API Host，此时旧域名会返回
    `403 Invalid Host`。GeoAPI 在新 Host 下的路径为 /geo/v2，
    同时保留 /v2 作为备选，调用时逐个尝试。

    Returns:
        (GeoAPI 候选基地址列表, 天气/指数 API 基地址)
    """
    host = (getattr(settings, "qweather_api_host", "") or "").strip()
    if not host:
        return ["https://geoapi.qweather.com/v2"], "https://devapi.qweather.com/v7"

    if not host.startswith("http"):
        host = f"https://{host}"
    host = host.rstrip("/")
    return [f"{host}/geo/v2", f"{host}/v2"], f"{host}/v7"


QWATHER_GEO_BASES, QWATHER_WEATHER_BASE = _resolve_qweather_bases()

# 西安 LocationID（和风天气城市编码）
XI_AN_LOCATION_ID = "101110101"

# 常用城市名 → LocationID 映射（减少 API 调用）
CITY_LOCATION_CACHE: Dict[str, str] = {
    "西安": XI_AN_LOCATION_ID,
    "咸阳": "101110200",
    "宝鸡": "101110900",
    "渭南": "101110500",
    "汉中": "101110800",
    "延安": "101110300",
    "榆林": "101110400",
    "铜川": "101111000",
    "安康": "101110700",
    "商洛": "101110600",
}


@dataclass
class CurrentWeather:
    """实时天气数据结构。"""

    city: str = ""
    temp: str = ""           # 温度（℃）
    feels_like: str = ""     # 体感温度（℃）
    text: str = ""           # 天气状况（晴/多云/雨等）
    wind_dir: str = ""       # 风向
    wind_scale: str = ""     # 风力等级
    humidity: str = ""       # 相对湿度（%）
    visibility: str = ""     # 能见度（km）
    pressure: str = ""       # 气压（hPa）
    obs_time: str = ""       # 观测时间

    def to_display(self) -> str:
        return (
            f"【{self.city}实时天气】\n"
            f"🌡 温度: {self.temp}℃（体感 {self.feels_like}℃）\n"
            f"☁ 天气: {self.text} | 💨 {self.wind_dir} {self.wind_scale}级\n"
            f"💧 湿度: {self.humidity}% | 👁 能见度: {self.visibility}km"
        )


@dataclass
class DailyForecast:
    """单日天气预报。"""

    date: str = ""
    temp_max: str = ""
    temp_min: str = ""
    text_day: str = ""       # 白天天气
    text_night: str = ""     # 夜间天气
    wind_dir_day: str = ""
    wind_scale_day: str = ""
    humidity: str = ""
    sunrise: str = ""
    sunset: str = ""

    def to_display(self) -> str:
        return (
            f"📅 {self.date}: {self.text_day}转{self.text_night} "
            f"🌡{self.temp_min}~{self.temp_max}℃ "
            f"💨{self.wind_dir_day}{self.wind_scale_day}级 "
            f"🌅{self.sunrise} 🌇{self.sunset}"
        )


@dataclass
class LifeIndex:
    """生活指数。"""

    name: str = ""
    level: str = ""
    category: str = ""
    text: str = ""


class QWeatherClient:
    """和风天气 API 客户端。

    使用方式：
        client = QWeatherClient(api_key=settings.qweather_api_key)
        weather = client.get_current_weather("西安")
        forecast = client.get_7day_forecast("西安")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "qweather_api_key", "")
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
    # 城市搜索
    # ------------------------------------------------------------------

    def _get_location_id(self, city: str) -> Optional[str]:
        """获取城市的 LocationID。

        优先从本地缓存取，缓存 miss 时调 API。
        GeoAPI 基地址可能有多个候选（新版专属 Host 使用 /geo/v2 路径），
        逐个尝试直到成功。
        """
        if city in CITY_LOCATION_CACHE:
            return CITY_LOCATION_CACHE[city]

        last_error = ""
        for base in QWATHER_GEO_BASES:
            try:
                resp = self.client.get(
                    f"{base}/city/lookup",
                    params={"location": city, "key": self.api_key},
                )
                if resp.status_code != 200:
                    last_error = f"{base} -> HTTP {resp.status_code}"
                    continue
                data = resp.json()
                if data.get("code") == "200" and data.get("location"):
                    loc_id = data["location"][0]["id"]
                    CITY_LOCATION_CACHE[city] = loc_id
                    return loc_id
                last_error = f"{base} -> {data}"
            except Exception as e:
                last_error = f"{base} -> {type(e).__name__}: {e}"

        logger.warning(f"[QWeather] 城市「{city}」未找到 LocationID: {last_error}")
        return None

    # ------------------------------------------------------------------
    # 实时天气
    # ------------------------------------------------------------------

    def get_current_weather(self, city: str = "西安") -> Optional[CurrentWeather]:
        """获取城市实时天气。

        Returns:
            CurrentWeather 对象，失败返回 None
        """
        loc_id = self._get_location_id(city)
        if not loc_id:
            return None

        try:
            resp = self.client.get(
                f"{QWATHER_WEATHER_BASE}/weather/now",
                params={"location": loc_id, "key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "200":
                logger.warning(f"[QWeather] 实时天气查询失败: {data}")
                return None

            now = data["now"]
            return CurrentWeather(
                city=city,
                temp=now.get("temp", ""),
                feels_like=now.get("feelsLike", ""),
                text=now.get("text", ""),
                wind_dir=now.get("windDir", ""),
                wind_scale=now.get("windScale", ""),
                humidity=now.get("humidity", ""),
                visibility=now.get("vis", ""),
                pressure=now.get("pressure", ""),
                obs_time=now.get("obsTime", ""),
            )
        except Exception as e:
            logger.error(f"[QWeather] 获取实时天气异常: {e}")
            return None

    # ------------------------------------------------------------------
    # 7天预报
    # ------------------------------------------------------------------

    def get_7day_forecast(self, city: str = "西安") -> List[DailyForecast]:
        """获取城市7天天气预报。

        Returns:
            DailyForecast 列表，失败返回空列表
        """
        loc_id = self._get_location_id(city)
        if not loc_id:
            return []

        try:
            resp = self.client.get(
                f"{QWATHER_WEATHER_BASE}/weather/7d",
                params={"location": loc_id, "key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "200":
                logger.warning(f"[QWeather] 7天预报查询失败: {data}")
                return []

            forecasts: List[DailyForecast] = []
            for d in data.get("daily", []):
                forecasts.append(DailyForecast(
                    date=d.get("fxDate", ""),
                    temp_max=d.get("tempMax", ""),
                    temp_min=d.get("tempMin", ""),
                    text_day=d.get("textDay", ""),
                    text_night=d.get("textNight", ""),
                    wind_dir_day=d.get("windDirDay", ""),
                    wind_scale_day=d.get("windScaleDay", ""),
                    humidity=d.get("humidity", ""),
                    sunrise=d.get("sunrise", ""),
                    sunset=d.get("sunset", ""),
                ))
            return forecasts
        except Exception as e:
            logger.error(f"[QWeather] 获取7天预报异常: {e}")
            return []

    # ------------------------------------------------------------------
    # 生活指数
    # ------------------------------------------------------------------

    def get_life_indices(self, city: str = "西安", index_type: str = "0") -> List[LifeIndex]:
        """获取生活指数。

        Args:
            city: 城市名
            index_type: 指数类型
                0=全部, 1=运动, 2=洗车, 3=穿衣, 5=紫外线, 7=旅游, 8=感冒

        Returns:
            LifeIndex 列表
        """
        loc_id = self._get_location_id(city)
        if not loc_id:
            return []

        try:
            resp = self.client.get(
                f"{QWATHER_WEATHER_BASE}/indices/1d",
                params={
                    "location": loc_id,
                    "key": self.api_key,
                    "type": index_type,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "200":
                logger.warning(f"[QWeather] 生活指数查询失败: {data}")
                return []

            indices: List[LifeIndex] = []
            for d in data.get("daily", []):
                indices.append(LifeIndex(
                    name=d.get("name", ""),
                    level=d.get("level", ""),
                    category=d.get("category", ""),
                    text=d.get("text", ""),
                ))
            return indices
        except Exception as e:
            logger.error(f"[QWeather] 获取生活指数异常: {e}")
            return []


# ==================== 简单内存缓存 ====================

@dataclass
class _CacheEntry:
    data: Any
    expire_at: float


class WeatherCache:
    """天气数据简单缓存（TTL 5分钟）。"""

    def __init__(self, ttl: float = 300.0):
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

    def clear(self):
        self._store.clear()


_weather_cache = WeatherCache(ttl=300.0)


def get_weather_client() -> QWeatherClient:
    return QWeatherClient()


def get_cached_current_weather(city: str = "西安") -> Optional[CurrentWeather]:
    """获取实时天气（带缓存）。"""
    cache_key = f"now:{city}"
    cached = _weather_cache.get(cache_key)
    if cached is not None:
        return cached

    client = get_weather_client()
    try:
        result = client.get_current_weather(city)
        _weather_cache.set(cache_key, result)
        return result
    finally:
        client.close()


def get_cached_forecast(city: str = "西安") -> List[DailyForecast]:
    """获取7天预报（带缓存）。"""
    cache_key = f"forecast:{city}"
    cached = _weather_cache.get(cache_key)
    if cached is not None:
        return cached

    client = get_weather_client()
    try:
        result = client.get_7day_forecast(city)
        _weather_cache.set(cache_key, result)
        return result
    finally:
        client.close()


def get_cached_life_indices(city: str = "西安", index_type: str = "3") -> List[LifeIndex]:
    """获取生活指数（带缓存）。

    Args:
        city: 城市名
        index_type: 指数类型，3=穿衣指数
    """
    cache_key = f"indices:{city}:{index_type}"
    cached = _weather_cache.get(cache_key)
    if cached is not None:
        return cached

    client = get_weather_client()
    try:
        result = client.get_life_indices(city, index_type)
        _weather_cache.set(cache_key, result)
        return result
    finally:
        client.close()


def format_weather_for_agent(city: str = "西安", date: str = "today") -> str:
    """格式化天气数据为 Agent 可读的 JSON 字符串。

    Args:
        city: 城市名
        date: 日期（"today" / "tomorrow" / 具体日期如 "2026-11-08"）

    Returns:
        JSON 字符串
    """
    import json as _json
    from datetime import date as _date, timedelta

    result: Dict[str, Any] = {"city": city, "date": date}

    forecasts = get_cached_forecast(city)

    # 解析目标日期
    if date in ("today", "今天", ""):
        target = _date.today().isoformat()
        is_future = False
    elif date in ("tomorrow", "明天"):
        target = (_date.today() + timedelta(days=1)).isoformat()
        is_future = True
    elif date in ("day_after_tomorrow", "后天"):
        target = (_date.today() + timedelta(days=2)).isoformat()
        is_future = True
    else:
        try:
            target = _date.fromisoformat(str(date).strip()).isoformat()
        except (ValueError, TypeError):
            target = _date.today().isoformat()
        is_future = target != _date.today().isoformat()

    target_forecast = next((f for f in forecasts if f.date == target), None)

    # 实时天气（仅今天）
    if not is_future:
        current = get_cached_current_weather(city)
        if current is None and target_forecast is None:
            return _json.dumps({
                "error": f"暂未查询到「{city}」的天气信息，请稍后重试或换个城市试试",
            }, ensure_ascii=False)
        if current is not None:
            result["current"] = {
                "temp": current.temp,
                "feels_like": current.feels_like,
                "text": current.text,
                "wind_dir": current.wind_dir,
                "wind_scale": current.wind_scale,
                "humidity": current.humidity,
            }

    # 目标日期预报
    if target_forecast is not None:
        result["target_forecast"] = {
            "date": target_forecast.date,
            "temp_max": target_forecast.temp_max,
            "temp_min": target_forecast.temp_min,
            "text_day": target_forecast.text_day,
            "text_night": target_forecast.text_night,
            "wind": f"{target_forecast.wind_dir_day}{target_forecast.wind_scale_day}级",
        }
    elif is_future:
        return _json.dumps({
            "error": f"未查到「{city}」{target} 的天气预报，仅支持未来7天",
        }, ensure_ascii=False)

    # 3天概览
    result["forecast"] = [{
        "date": f.date,
        "temp_max": f.temp_max,
        "temp_min": f.temp_min,
        "text_day": f.text_day,
        "text_night": f.text_night,
        "wind": f"{f.wind_dir_day}{f.wind_scale_day}级",
    } for f in forecasts[:3]]

    # 穿衣建议：优先用官方穿衣指数，失败时按温度兜底
    indices = get_cached_life_indices(city, "3")
    if indices:
        idx = indices[0]
        result["dress_suggestion"] = f"{idx.category}，{idx.text}"
    else:
        temp_ref = None
        if target_forecast is not None:
            try:
                temp_ref = (
                    float(target_forecast.temp_max) + float(target_forecast.temp_min)
                ) / 2
            except (ValueError, TypeError):
                temp_ref = None
        if temp_ref is None and result.get("current", {}).get("temp"):
            try:
                temp_ref = float(result["current"]["temp"])
            except (ValueError, TypeError):
                temp_ref = None

        if temp_ref is not None:
            if temp_ref < 5:
                dress = "羽绒服/厚外套"
            elif temp_ref < 12:
                dress = "厚外套/毛衣"
            elif temp_ref < 22:
                dress = "长袖衬衫/薄毛衣"
            else:
                dress = "短袖/裙子"
            result["dress_suggestion"] = f"建议穿{dress}"

    # 天气提醒（区分今天/未来，避免说错日期）
    day_label = "今天" if not is_future else target[5:]
    if is_future and target_forecast is not None:
        weather_text = f"{target_forecast.text_day}转{target_forecast.text_night}"
    else:
        weather_text = result.get("current", {}).get("text", "")

    if weather_text and ("雨" in weather_text or "雪" in weather_text):
        result["reminder"] = f"⚠️ {day_label}{weather_text}，建议携带雨具"
    elif weather_text:
        result["reminder"] = f"{day_label}天气{weather_text}"

    return _json.dumps(result, ensure_ascii=False)