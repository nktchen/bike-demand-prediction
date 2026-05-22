import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODELS_DIR,
    MODEL_PATH,
    RAW_DATA_DIR,
    DEMO_DATA_DIR,
    
)

from src.features import (
    CAT_FEATURES,
    ALL_FEATURES,
    add_rush_hour_features, 
    add_time_features,
    add_context_features,
    add_lag_features,
)

from src.config import RANDOM_STATE

CATBOOST_PARAMS = {
    "iterations": 1580,
    "learning_rate": 0.058755601353182224,
    "depth": 5,
    "l2_leaf_reg": 2.8838758703112877,
    "random_strength": 1.4654473731747668,
    "bagging_temperature": 1.3682099526511078,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
}

def download_data() -> pd.DataFrame:
    raw_csv_path = RAW_DATA_DIR / "hour.csv"
    if raw_csv_path.exists():
        return pd.read_csv(raw_csv_path)

    import kagglehub
    print("starting downloading")
    dataset_path = Path(kagglehub.dataset_download("marklvl/bike-sharing-dataset"))
    downloaded_csv_path = dataset_path / "hour.csv"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_csv_path, raw_csv_path)
    return pd.read_csv(raw_csv_path)

def add_timestamp_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = (
        pd.to_datetime(df["dteday"]) +
        pd.to_timedelta(df["hr"], unit="h")
    )
    return df.sort_values("timestamp").reset_index(drop=True)

def prepare_training_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    #исторические данные - сортируем, чтобы не было утечки данных из будущего.
    df = add_timestamp_and_sort(df)
    df["weathersit"] = df["weathersit"].replace(4, 3)
    df = add_rush_hour_features(df)
    df = add_time_features(df)
    df = add_context_features(df)
    df = add_lag_features(df)

    return df

def split_data(df: pd.DataFrame, train_size_ratio = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_size = int(len(df) * train_size_ratio)
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return (train_df, test_df)

def train_catboost(train_data: pd.DataFrame) -> CatBoostRegressor:
    model = CatBoostRegressor(**CATBOOST_PARAMS)

    model.fit(
        train_data[ALL_FEATURES],
        train_data["cnt"],
        cat_features=CAT_FEATURES,
        verbose=False,
    )

    return model

def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = download_data()
    df_prepared = prepare_training_df(raw_df)
    train_df, test_df = split_data(df_prepared)
    
    test_metrics_model = train_catboost(train_df)
    test_metrics_pred = test_metrics_model.predict(test_df[ALL_FEATURES])
    test_metrics_metrics = regression_metrics(test_df["cnt"], test_metrics_pred)

    print("holdout test metrics:")
    print(json.dumps(test_metrics_metrics, indent=2))

    final_model = train_catboost(df_prepared)
    final_model.save_model(MODEL_PATH)

    print(f"saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()




