"""天气查询工具（基于 Open-Meteo）。"""

import time
from typing import Any

import httpx
from loguru import logger

from config.settings import get_settings
from tools.base import BaseTool
from tools.base import ToolResult
from utils.exception import ToolException

# 常见中国城市坐标
CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "成都": (30.5728, 104.0668),
    "杭州": (30.2741, 120.1551),
    "武汉": (30.5928, 114.3055),
    "南京": (32.0603, 118.7969),
    "重庆": (29.4316, 106.9123),
    "西安": (34.3416, 108.9398),
    "天津": (39.3434, 117.3616),
    "苏州": (31.2990, 120.5853),
    "长沙": (28.2282, 112.9388),
    "郑州": (34.7466, 113.6253),
    "东莞": (23.0208, 113.7518),
    "青岛": (36.0671, 120.3826),
    "沈阳": (41.8057, 123.4315),
    "宁波": (29.8683, 121.5440),
    "昆明": (25.0389, 102.7183),
    "大连": (38.9140, 121.6147),
    "厦门": (24.4798, 118.0894),
    "合肥": (31.8206, 117.2272),
    "佛山": (23.0215, 113.1214),
    "福州": (26.0745, 119.2965),
    "哈尔滨": (45.8038, 126.5350),
    "济南": (36.6512, 116.9972),
    "温州": (28.0015, 120.6994),
    "长春": (43.8171, 125.3235),
    "石家庄": (38.0428, 114.5149),
    "常州": (31.7712, 119.9740),
    "泉州": (24.8739, 118.6757),
    "南宁": (22.8170, 108.3665),
    "贵阳": (26.6470, 106.6302),
    "南昌": (28.6829, 115.8581),
    "太原": (37.8706, 112.5489),
    "烟台": (37.4635, 121.4479),
    "嘉兴": (30.7710, 120.7551),
    "南通": (31.9796, 120.8939),
    "金华": (29.1045, 119.6489),
    "珠海": (22.2710, 113.5667),
    "惠州": (23.1118, 114.4160),
    "徐州": (34.2057, 117.2841),
    "海口": (20.0440, 110.3497),
    "乌鲁木齐": (43.8256, 87.6168),
    "拉萨": (29.6500, 91.1000),
    "兰州": (36.0611, 103.8343),
    "呼和浩特": (40.8422, 111.7498),
}

# WMO 天气代码 → 中文描述
WMO_CODE_MAP: dict[int, str] = {
    0: "晴天",
    1: "大部晴朗",
    2: "多云",
    3: "阴天",
    45: "雾",
    48: "冰雾",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    56: "小冻毛毛雨",
    57: "大冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小冻雨",
    67: "大冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}

# 风向角度 → 中文方向
WIND_DIR_MAP: list[tuple[float, str]] = [
    (348.75, "北"),
    (326.25, "西北"),
    (303.75, "西"),
    (281.25, "西南"),
    (258.75, "南"),
    (236.25, "东南"),
    (213.75, "东"),
    (191.25, "东北"),
    (168.75, "北"),
    (146.25, "西北"),
    (123.75, "西"),
    (101.25, "西南"),
    (78.75, "南"),
    (56.25, "东南"),
    (33.75, "东"),
    (11.25, "东北"),
]


def _wmo_to_desc(code: int) -> str:
    return WMO_CODE_MAP.get(code, f"未知({code})")


def _degrees_to_wind_dir(degrees: float) -> str:
    """将角度转为中文风向。"""
    for threshold, name in WIND_DIR_MAP:
        if degrees >= threshold:
            return name
    return "北"


class WeatherTool(BaseTool):
    """查询指定城市的实时天气和未来 3 天天气预报。"""

    name = "weather"
    description = "查询指定城市的实时天气和未来 3 天天气预报。"
    input_schema = {"city": "必填，城市名称，例如 北京、上海。"}

    required_permissions = frozenset({"network:read"})

    def _resolve_coords(self, client: httpx.Client, city: str) -> tuple[float, float]:
        """根据城市名解析经纬度。"""
        coords = CITY_COORDS.get(city)
        if coords is not None:
            logger.info("[天气] 命中本地坐标表: {} -> {}, {}", city, coords[0], coords[1])
            return coords
        logger.info("[天气] 未命中本地坐标表，尝试 geocoding: {}", city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh"
        geo_resp = client.get(geo_url, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            raise ToolException(f"未找到城市「{city}」的坐标信息")
        lat = float(results[0]["latitude"])
        lon = float(results[0]["longitude"])
        logger.info("[天气] geocoding 结果: {} -> {}, {}", city, lat, lon)
        return lat, lon

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """查询指定城市的天气。

        参数:
            tool_input: 包含 `city` 的字典。

        返回:
            天气信息文本。

        异常:
            ToolException: 城市名为空或 HTTP 请求失败时抛出。
        """
        city = str(tool_input.get("city", "")).strip()
        if not city:
            raise ToolException("城市名称不能为空")
        settings = get_settings()

        # 瞬态网络错误重试（最多 3 次，指数退避）
        last_exc: Exception | None = None
        response = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=settings.request_timeout) as client:
                    lat, lon = self._resolve_coords(client, city)
                    url = (
                        f"https://api.open-meteo.com/v1/forecast"
                        f"?latitude={lat}&longitude={lon}"
                        f"&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
                        f"weather_code,wind_speed_10m,wind_direction_10m"
                        f"&daily=temperature_2m_max,temperature_2m_min,weather_code"
                        f"&timezone=auto&forecast_days=4"
                    )
                    logger.info("[天气] 请求 Open-Meteo: lat={} lon={}", lat, lon)
                    response = client.get(url, headers={"User-Agent": "AgentOffice/1.0"})
                    logger.info(
                        "[天气] status={} body_len={}",
                        response.status_code,
                        len(response.content),
                    )
                    response.raise_for_status()
                    last_exc = None
                    break
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 0.5 * (2**attempt)
                    logger.warning("[天气] 第{}次请求失败，{}s后重试: {}", attempt + 1, wait, exc)
                    time.sleep(wait)
                continue

        if last_exc is not None:
            raise ToolException(f"天气查询失败：{last_exc}") from last_exc

        data = response.json()
        current = data.get("current")
        if not current:
            raise ToolException("天气 API 未返回当前天气数据，请稍后重试")

        temp = str(current.get("temperature_2m", "N/A"))
        feels_like = str(current.get("apparent_temperature", "N/A"))
        humidity = str(current.get("relative_humidity_2m", "N/A"))
        weather_code = int(current.get("weather_code", 0))
        desc = _wmo_to_desc(weather_code)
        wind_speed = str(current.get("wind_speed_10m", "N/A"))
        wind_dir = _degrees_to_wind_dir(float(current.get("wind_direction_10m", 0)))

        forecasts: list[str] = []
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        wmo_codes = daily.get("weather_code", [])
        for i in range(min(len(dates), 3)):
            day_desc = _wmo_to_desc(int(wmo_codes[i]) if i < len(wmo_codes) else 0)
            forecasts.append(
                f"  {dates[i]}: {min_temps[i]}~{max_temps[i]}°C, {day_desc}"
            )

        parts = [
            f"{city} 当前天气：{desc}，{temp}°C（体感 {feels_like}°C），"
            f"湿度 {humidity}%，{wind_dir} 风 {wind_speed}km/h",
        ]
        if forecasts:
            parts.append("未来 3 天预报：\n" + "\n".join(forecasts))
        content = "\n\n".join(parts)
        return ToolResult(
            content=content,
            metadata={"city": city, "temperature": temp, "condition": desc},
        )
