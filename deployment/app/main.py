from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import os
from contextlib import asynccontextmanager

model = None

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
    data = await request.json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
