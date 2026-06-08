"""WeatherTool regression tests."""

from tools.base import ToolResult
from tools.weather_tool import WeatherTool


class _FakeWeatherResponse:
    status_code = 200
    text = "{}"
    content = b"{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "current": {
                "temperature_2m": 25,
                "apparent_temperature": 26,
                "relative_humidity_2m": 50,
                "weather_code": 0,
                "wind_speed_10m": 3,
                "wind_direction_10m": 90,
            },
            "daily": {
                "time": ["2026-06-02"],
                "temperature_2m_max": [30],
                "temperature_2m_min": [20],
                "weather_code": [0],
            },
        }


class _FakeHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False

    def get(self, *args, **kwargs) -> _FakeWeatherResponse:
        return _FakeWeatherResponse()


def test_weather_tool_returns_tool_result(monkeypatch) -> None:
    """A successful weather lookup must return ToolResult, not None."""
    import tools.weather_tool as weather_module

    monkeypatch.setattr(weather_module.httpx, "Client", _FakeHttpxClient)
    monkeypatch.setattr(
        WeatherTool,
        "_resolve_coords",
        lambda self, client, city: (39.9, 116.4),
    )

    result = WeatherTool().run({"city": "beijing"})

    assert isinstance(result, ToolResult)
    assert result.content
    assert result.metadata["city"] == "beijing"
