import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import PredictionRequest, PredictionResponse
from app.model_service import ml_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_logger")

MODEL_PATH = "model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_service.load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    yield
    # Clean up resources if necessary (shutdown logic)

app = FastAPI(
    title="Iris Classification API",
    version="1.0.0",
  description="A containerized REST API for serving scikit-learn classification models using FastAPI and Docker.",
    lifespan=lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if not ml_service.model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "service": "iris-classifier"}

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: PredictionRequest):
    try:
        result = ml_service.predict(request.features)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction")