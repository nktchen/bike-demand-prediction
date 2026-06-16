import httpx
import logging
logger = logging.getLogger("bike_demand_api")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PARAMS = {
    "latitude": 38.8951,
    "longitude": -77.0364,
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "weather_code,"
        "apparent_temperature,"
        "wind_speed_10m"
    ),
    "forecast_days": 1,
    "timezone": "America/New_York",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
}
DEFAULT_TIMESTAMP = "2012-12-28T_:00:00"
DEFAULT_HOLIDAY = 0

def open_meteo_code_to_weathersit(weather_code: int) -> int:
    if weather_code in (0, 1):
        return 1
    if weather_code in (2, 3, 45, 48):
        return 2
    return 3

def get_default_time_data(current_hour: int) -> dict[str, int | str]:
    return {
        "timestamp": DEFAULT_TIMESTAMP.replace("_", f"{current_hour:02d}"), #adds current hour to hardcoded default timestamp
        "holiday": DEFAULT_HOLIDAY,
    }

async def fetch_weather_from_open_meteo() -> dict[str, float | int | str]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            logger.info(
                "open_meteo_request url=%s params=%s",
                OPEN_METEO_URL,
                OPEN_METEO_PARAMS,
            )
            response = await client.get(
                OPEN_METEO_URL,
                params=OPEN_METEO_PARAMS,
            )
            logger.info(
                "open_meteo_response status_code=%s url=%s",
                response.status_code,
                response.url,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        response = getattr(exc, "response", None)

        logger.exception(
            "open_meteo_error type=%s url=%s status_code=%s response_body=%r exception=%r",
            type(exc).__name__,
            getattr(exc.request, "url", OPEN_METEO_URL),
            response.status_code if response is not None else None,
            response.text[:1000] if response is not None else None,
            exc,
        )
        raise
    try:
        payload = response.json()

        hourly = payload["hourly"]
        row_index = 0

        weather_code = int(hourly["weather_code"][row_index])

        return {
            "temp_c": float(hourly["temperature_2m"][row_index]),
            "feels_like_c": float(hourly["apparent_temperature"][row_index]),
            "humidity_percent": float(hourly["relative_humidity_2m"][row_index]),
            "wind_speed_kmh": float(hourly["wind_speed_10m"][row_index]),
            "weathersit": open_meteo_code_to_weathersit(weather_code),
        }
    except (ValueError, TypeError, KeyError, IndexError):
        logger.exception(
            "open_meteo_invalid_response response_body=%r",
            response.text[:2000],
        )

    raise