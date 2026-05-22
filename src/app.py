from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.predict import load_demo_history, load_model, predict_one
from src.schemas import HealthResponse, PredictRequest, PredictResponse


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
        return predict_one(
            model=model,
            request=request,
            history=demo_history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

