import json

from src.config import (
    DEMO_DATA_DIR,
    DEMO_HISTORY_PATH,
    DEMO_REQUESTS_PATH,
    RAW_DATA_DIR,
    RANDOM_STATE,

)
from src.features import denormalize_weather
from src.train import download_data, add_timestamp_and_sort


def main() -> None:
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = download_data()
    data = add_timestamp_and_sort(data)
    data["weathersit"] = data["weathersit"].replace(4, 3)
    demo_history = data[["timestamp", "cnt"]].copy()
    demo_history.to_csv(DEMO_HISTORY_PATH, index=False)

    demo_source = data.iloc[-100:].sample(10, random_state=RANDOM_STATE)

    demo_requests = []
    for _, row in demo_source.iterrows():
        weather = denormalize_weather(row)
        demo_requests.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "temp_c": weather["temp_c"],
                "feels_like_c": weather["feels_like_c"],
                "humidity_percent": weather["humidity_percent"],
                "wind_speed_kmh": weather["wind_speed_kmh"],
                "weathersit": int(row["weathersit"]),
                "holiday": int(row["holiday"]),
                "actual_cnt": int(row["cnt"]),
            }
        )
    DEMO_REQUESTS_PATH.write_text(
        json.dumps(demo_requests, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved demo history: {DEMO_HISTORY_PATH}")
    print(f"saved demo requests: {DEMO_REQUESTS_PATH}")

if __name__ == "__main__":
    main()
