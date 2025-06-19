from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import os
from contextlib import asynccontextmanager
import time
import logging
from fastapi.responses import PlainTextResponse
import joblib

metrics = {"total_predictions": 0}

model = None
scaler = None
encoders = None

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoders
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise ValueError("MODEL_URI environment variable is not set.")
    model = mlflow.pyfunc.load_model(model_uri)

    # Step 1: Get the model version's run ID (if using registry URI)
    if model_uri.startswith("models:/"):
        client = mlflow.MlflowClient()
        name, alias = model_uri.replace("models:/", "").split("@")
        version_info = client.get_model_version_by_alias(name, alias)
        run_id = version_info.run_id
    elif model_uri.startswith("runs:/"):
        run_id = model_uri.split("/")[1]
    else:
        raise ValueError("Unsupported MODEL_URI format.")

    # Step 2: Download artifacts from that run
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/scaler.pkl")
    encoders_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/encoders.pkl")

    # Step 3: Load them into memory
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    global model, scaler, encoders
    start = time.time()
    data = await request.json()
    df = pd.DataFrame([data])

    # Apply label encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict using MLflow model
    prediction = model.predict(df_scaled)
    duration = time.time() - start
    metrics["total_predictions"] += 1
    logging.info(f"Prediction: input={data}, output={prediction.tolist()}, time={duration:.3f}s")
    
    return {"prediction": prediction.tolist()}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    return f'total_predictions {metrics["total_predictions"]}\n'