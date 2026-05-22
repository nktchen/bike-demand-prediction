from datetime import datetime
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    timestamp: datetime
    temp_c: float #not normalisied
    feels_like_c: float #not normalisied
    humidity_percent: float = Field(ge=0, le=100)
    wind_speed_kmh: float = Field(ge=0)
    weathersit: int = Field(ge=1, le=3)
    holiday: int = Field(ge=0, le=1)

class PredictResponse(BaseModel):
    prediction: float
    prediction_rounded: int
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
