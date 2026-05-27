import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast?latitude=38.8951&longitude=-77.0364&hourly=temperature_2m,relative_humidity_2m,weather_code,apparent_temperature,wind_speed_10m&forecast_days=1&timezone=America%2FNew_York&temperature_unit=celsius&wind_speed_unit=kmh"
DEFAULT_TIMESTAMP = "2012-12-28T_:00:00"
DEFAULT_HOLIDAY = 0

def open_meteo_code_to_weathersit(weather_code: int) -> int:
    if weather_code in (0, 1):
        return 1
    if weather_code in (2, 3, 45, 48):
        return 2
    return 3

def get_default_time_data(cur_datetime: int) -> dict[str, int | str]:
    return {
        "timestamp": DEFAULT_TIMESTAMP.replace("_", str(cur_datetime)), #adds current hour to hardcoded default timestamp
        "holiday": DEFAULT_HOLIDAY,
    }

async def fetch_weather_from_open_meteo() -> dict[str, float | int | str]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(OPEN_METEO_URL)
    response.raise_for_status()
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
