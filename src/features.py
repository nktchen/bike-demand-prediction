import numpy as np
import pandas as pd

from src.config import WEATHER_NORMALISATION_BORDERS, YR_DEMO_VALUE
from src.schemas import PredictRequest

CAT_FEATURES = [
    "yr",
    "season",
    "mnth",
    "hr",
    "weekday",
    "holiday",
    "workingday",
    "weathersit",
]

NUM_FEATURES = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


EXTRA_FEATURES = [
    "hr_sin",
    "hr_cos",
    "weekday_sin",
    "weekday_cos",
    "mnth_sin",
    "mnth_cos",
    "is_working_day_rush",
    "is_weekend_day_rush",
    "is_night",
    "is_cold_season",
    "is_good_weather",
    "is_bad_weather",
    "is_comfortable_weather",
    "is_high_humidity",
    "is_high_wind",
    "cnt_lag_24h",
    "cnt_lag_168h",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + EXTRA_FEATURES
HISTORY_FEATURES = ["cnt_lag_24h", "cnt_lag_168h"]

def add_rush_hour_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    working_day_rush_hours = [8, 17, 18]
    weekend_day_rush_hours = [12,13,14,15,16]

    data['is_working_day_rush'] = ((data['workingday'] == 1) & data['hr'].isin(working_day_rush_hours)).astype(int)
    data['is_weekend_day_rush'] = ((data['workingday'] == 0) & data['hr'].isin(weekend_day_rush_hours)).astype(int)
    return data

def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["hr_sin"] = np.sin(2 * np.pi * data["hr"] / 24)
    data["hr_cos"] = np.cos(2 * np.pi * data["hr"] / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
    data["mnth_sin"] = np.sin(2 * np.pi * (data["mnth"] - 1) / 12)
    data["mnth_cos"] = np.cos(2 * np.pi * (data["mnth"] - 1) / 12)

    return data

def add_context_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["is_night"] = data["hr"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    data["is_warm_season"] = data["mnth"].isin([5, 6, 7, 8, 9]).astype(int)
    data["is_cold_season"] = data["mnth"].isin([12, 1, 2]).astype(int)

    data["is_good_weather"] = (data["weathersit"] == 1).astype(int)
    data["is_bad_weather"] = (data["weathersit"] >= 3).astype(int)

    data["is_comfortable_weather"] = (
        (data["temp"].between(0.4, 0.8)) &
        (data["hum"] < 0.8) &
        (data["windspeed"] < 0.4) &
        (data["weathersit"] <= 2)
    ).astype(int)

    data["is_high_humidity"] = (data["hum"] > 0.8).astype(int)
    data["is_high_wind"] = (data["windspeed"] > 0.4).astype(int)
    return data

def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["cnt_lag_24h"] = data["cnt"].shift(24)
    data["cnt_lag_168h"] = data["cnt"].shift(168)
    data = data.dropna(subset=HISTORY_FEATURES).copy()
    return data

def get_feature_borders() -> dict[str, int]:
    norm_borders : dict[str, int] = {}
    for weather_feature_name, low_up_borders in WEATHER_NORMALISATION_BORDERS.items():
        norm_borders[f"{weather_feature_name}_low_border"] = low_up_borders[0]
        norm_borders[f"{weather_feature_name}_up_border"] = low_up_borders[1]
    return norm_borders

def normalize_weather(
    temp_c: float,
    feels_like_c: float,
    humidity_percent: float,
    wind_speed_kmh: float,
) -> dict[str, float]:
    norm_borders = get_feature_borders()
    normalized_weather = {
        "temp": (temp_c + norm_borders["temp_low_border"]) / norm_borders["temp_up_border"],
        "atemp": (feels_like_c + norm_borders["atemp_low_border"]) / norm_borders["atemp_up_border"],
        "hum": (humidity_percent + norm_borders["hum_low_border"]) / norm_borders["hum_up_border"],
        "windspeed": (wind_speed_kmh + norm_borders["windspeed_low_border"]) / norm_borders["windspeed_up_border"],
    }
    for weather_feature_name, value in normalized_weather.items():
        if value < 0:
            normalized_weather[weather_feature_name] = 0
        elif value > 1:
            normalized_weather[weather_feature_name] = 1
    return normalized_weather


def denormalize_weather(row: pd.Series) -> dict[str, float]:
    norm_borders = get_feature_borders()
    return {
        "temp_c": float(row["temp"] * norm_borders["temp_up_border"]) - norm_borders["temp_low_border"],
        "feels_like_c": float(row["atemp"] * norm_borders["atemp_up_border"]) - norm_borders["atemp_low_border"],
        "humidity_percent": float(row["hum"] * norm_borders["hum_up_border"]) - norm_borders["hum_low_border"],
        "wind_speed_kmh": float(row["windspeed"] * norm_borders["windspeed_up_border"]) - norm_borders["windspeed_low_border"],
    }

def month_to_season(month: int) -> int:
    if month in (1, 2, 3):
        return 1
    if month in (4, 5, 6):
        return 2
    if month in (7, 8, 9):
        return 3
    return 4

def ensure_feature_order(
    data: pd.DataFrame
) -> pd.DataFrame:
    missing_features = [feature for feature in ALL_FEATURES if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    return data[ALL_FEATURES]

def build_inference_features(
    request: PredictRequest,
    cnt_lag_24h: float,
    cnt_lag_168h: float,
) -> pd.DataFrame:
    timestamp = request.timestamp
    holiday = request.holiday
    weekday = timestamp.weekday()
    workingday = int((weekday not in (5, 6)) and holiday == 0)
    weather = normalize_weather(
        request.temp_c,
        request.feels_like_c,
        request.humidity_percent,
        request.wind_speed_kmh,
    )
    row = {
        "yr": YR_DEMO_VALUE,
        "season": month_to_season(timestamp.month),
        "mnth": timestamp.month,
        "hr": timestamp.hour,
        "weekday": weekday,
        "holiday": holiday,
        "workingday": workingday,
        "weathersit": request.weathersit,
        **weather,
        "cnt_lag_24h": cnt_lag_24h,
        "cnt_lag_168h": cnt_lag_168h,
    }
    features = pd.DataFrame([row])
    features = add_rush_hour_features(features)
    features = add_time_features(features)
    features = add_context_features(features)

    return ensure_feature_order(features)