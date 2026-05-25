import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.predict import load_demo_history, load_model, predict_one
from src.schemas import HealthResponse, PredictRequest, PredictResponse


LOG_PATH = Path(__file__).resolve().parents[1] / "logs.txt"

logger = logging.getLogger("bike_demand_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)


app = FastAPI(title="bike demand prediction API")

model = load_model()
demo_history = load_demo_history()

@app.get('/')
def root():
    return {"message": "hi! use /predict pls"}

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        response = predict_one(
            model=model,
            request=request,
            history=demo_history,
        )
        logger.info(
            "predict_ok request=%s response=%s",
            request.model_dump(mode="json"),
            response.model_dump(mode="json"),
        )
        return response
    except ValueError as exc:
        logger.warning(
            "predict_error request=%s error=%s",
            request.model_dump(mode="json"),
            str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
