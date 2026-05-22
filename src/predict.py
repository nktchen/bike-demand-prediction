import json
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

from src.config import DEMO_HISTORY_PATH, DEMO_REQUESTS_PATH, MODEL_PATH
from src.features import build_inference_features
from src.schemas import PredictRequest, PredictResponse

def load_model(path: Path = MODEL_PATH) -> CatBoostRegressor:
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}. run python -m src.train")
    model = CatBoostRegressor()
    model.load_model(path)
    return model

def load_demo_history(path: Path = DEMO_HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"demo history file not found: {path}. run python -m src.make_demo_data")
    history = pd.read_csv(path, parse_dates=["timestamp"])
    history = history.set_index("timestamp").sort_index()
    return history

def get_history_lag_value(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    hour: int,
) -> float | None:
    lag_timestamp = timestamp - pd.Timedelta(hours=hour)
    if lag_timestamp not in history.index:
        return None
    value = history.loc[lag_timestamp, "cnt"]
    return float(value) # type: ignore

def predict_one(
    model: CatBoostRegressor,
    request: PredictRequest,
    history: pd.DataFrame,
) -> PredictResponse:
    cnt_lag_24h = get_history_lag_value(history, pd.Timestamp(request.timestamp), 24)
    cnt_lag_168h = get_history_lag_value(history, pd.Timestamp(request.timestamp), 168)
    if cnt_lag_24h is None or cnt_lag_168h is None:
        raise ValueError(
            f"not enough historical data for {'cnt_lag_24h' if cnt_lag_24h is None else ''}/{'cnt_lag_168h' if cnt_lag_168h is None else ''}"
        )
    features = build_inference_features(
        request=request,
        cnt_lag_24h=cnt_lag_24h,
        cnt_lag_168h=cnt_lag_168h,
    )
    prediction = float(model.predict(features)[0])

    return PredictResponse(
        prediction=prediction,
        prediction_rounded=round(prediction),
        timestamp=request.timestamp,
    )

def load_request(path: Path, index: int) -> PredictRequest:
    if not path.exists():
        raise FileNotFoundError(f"request file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[index]
    return PredictRequest.model_validate(payload)

def main() -> None:
    request_number = int(input("select demo request - enter number from 1 to 10:\n")) - 1
    model = load_model()
    history = load_demo_history()
    request = load_request(DEMO_REQUESTS_PATH, request_number)
    response = predict_one(
        model=model,
        request=request,
        history=history,
    )
    print(response.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
