from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import os
from contextlib import asynccontextmanager
import time
import logging
from fastapi.responses import PlainTextResponse

metrics = {"total_predictions": 0}

model = None

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise ValueError("MODEL_URI environment variable is not set.")
    model = mlflow.pyfunc.load_model(model_uri)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    start = time.time()
    data = await request.json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    duration = time.time() - start
    metrics["total_predictions"] += 1
    logging.info(f"Prediction: input={data}, output={prediction.tolist()}, time={duration:.3f}s")
    
    return {"prediction": prediction.tolist()}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    return f'total_predictions {metrics["total_predictions"]}\n'