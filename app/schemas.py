from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        min_items=4, 
        max_items=4, 
        description="List of 4 numerical features: sepal_length, sepal_width, petal_length, petal_width",
        example=[5.1, 3.5, 1.4, 0.2]
    )

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence_score: float = Field(default=None, description="Probability of the predicted class")