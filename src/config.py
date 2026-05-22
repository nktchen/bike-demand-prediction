from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEMO_DATA_DIR = DATA_DIR / "demo"
DEMO_HISTORY_PATH = DEMO_DATA_DIR / "demo_history.csv"
DEMO_REQUESTS_PATH = DEMO_DATA_DIR / "demo_requests.json"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "catboost_model.cbm"

# normalisation borders are from original kaggle dataset - https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset
WEATHER_NORMALISATION_BORDERS : dict[str, tuple[int, int]] = {
        "temp": (-8, 39),
        "atemp": (-16, 50),
        "hum": (0, 100),
        "windspeed":  (0, 67)
}

YR_DEMO_VALUE = 1 #захардкодил признак yr, тк данные в обучающей выборке за 2011-2012 год. 1 значит 2012
RANDOM_STATE = 42