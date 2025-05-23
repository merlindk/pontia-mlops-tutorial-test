from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise ValueError("MODEL_URI environment variable is not set.")
    model = mlflow.pyfunc.load_model(model_uri)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
