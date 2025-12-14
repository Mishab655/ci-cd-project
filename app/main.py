"""FastAPI application entry point."""
from fastapi import FastAPI
from app.model import IrisModel
from pydantic import BaseModel

class IrisRequest(BaseModel):
    """Request schema for Iris prediction."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()
model = IrisModel()

@app.get("/")
def health():
    """Health check endpoint."""
    return {"status": "Iris ML API is running"}

@app.post("/predict")
def predict(request: IrisRequest):
    """Predict Iris flower class."""
    features = [
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]
    prediction = model.predict(features)
    return {"prediction": int(prediction)}
