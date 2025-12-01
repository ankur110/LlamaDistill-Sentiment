# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of input texts")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    calibrate: Optional[bool] = True
    calibrate_T: Optional[float] = 1.0

class SinglePrediction(BaseModel):
    pred: int
    probs: List[float]

class PredictResponse(BaseModel):
    results: List[SinglePrediction]
    model: str
    device: str
    latency_ms: float
