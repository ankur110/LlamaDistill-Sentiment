# src/api.py
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

from .inference import load_model_and_tokenizer, predict_texts
from .schemas import PredictRequest, PredictResponse, SinglePrediction
from .utils import format_results, now_ms
from .mlflow_utils import init_mlflow, log_inference_sample

# envs
MODEL_DIR = os.getenv("MODEL_DIR", "./outputs/mlflow_artifacts/student_model")
BASE_MODEL = os.getenv("BASE_MODEL", "NousResearch/Llama-3.2-1B")
MLFLOW_URI = os.getenv("MLFLOW_URI", "./outputs/mlflow_artifacts/inference")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "deployment-inference")

app = FastAPI(title="Sentiment Classifier (FastAPI)", version="0.1")


@app.on_event("startup")
def startup_event():
    global tokenizer, model, device
    tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR, BASE_MODEL)
   
    try:
        init_mlflow(mlflow_uri=MLFLOW_URI, experiment_name=MLFLOW_EXPERIMENT)
    except Exception:
        pass
@app.get("/")
def root():
    return {
        "message": "Sentiment Classifier API is live",
        "docs_url": "http://localhost:8000/docs",
        "version": "0.1"
    }

@app.get("/health")
def health():
    try:
        dev = device.type
    except Exception:
        dev = "unknown"
    return {"status": "ok", "device": dev, "model_loaded": True}

@app.get("/info")
def info():
    return {
        "model_dir": MODEL_DIR,
        "base_model": BASE_MODEL,
        "device": device.type,
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    texts = req.texts
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="`texts` is required and must be non-empty")

    t0 = now_ms()
    preds, probs, logits = predict_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16,
        max_length=128,
        calibrate_T=(req.calibrate_T if req.calibrate else None),
        threshold=req.threshold,
    )
    t1 = now_ms()

    results = [{"pred": int(p), "probs": pr} for p, pr in zip(preds, probs)]
    latency = float(t1 - t0)

    
    try:
        log_inference_sample(run_id="api", payload={"n": len(texts)}, preds=preds, probs=probs, meta={"latency_ms": latency})
    except Exception:
        pass

    return {
        "results": results,
        "model": MODEL_DIR,
        "device": device.type,
        "latency_ms": latency,
    }

# local run
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
