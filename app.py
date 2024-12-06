from pathlib import  Path
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
from typing import List
import pandas as pd


class InferenceInput(BaseModel):
    features: List[List[float]]

class InferenceOutput(BaseModel):
    predictions: List[int]

app = FastAPI()

MODEL_PATH = Path(__file__).parent.joinpath("model/logistic_regression_model.joblib").as_posix()
model = load(MODEL_PATH)


@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = pd.DataFrame(input_data.features)
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)