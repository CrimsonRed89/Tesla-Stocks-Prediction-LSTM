from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()

# venv\Scripts\activate.bat  
# pip install -r requirements.txt
# uvicorn app:app --reload 

# http://127.0.0.1:8000/docs


model_LSTM = joblib.load('C:\\Coding1\\Coding\\python\\Projects\\Tesla Stocks Prediction\\tesla_stock_model.pkl')
scaler = joblib.load('C:\\Coding1\\Coding\\python\\Projects\\Tesla Stocks Prediction\\scaler.pkl')

class StockInput(BaseModel):
    Close: List[float]  

@app.post("/predict/")
async def predict_stock(data: StockInput):

    if len(data.Close) != 60:
        return {"detail": "The 'Close' feature must contain exactly 60 values."}
    
    scaled_input = scaler.transform(np.array(data.Close).reshape(-1, 1))
    

    input_features = scaled_input.reshape(1, 60, 1)  
    

    prediction = model_LSTM.predict(input_features)

    predicted_value = scaler.inverse_transform(prediction.reshape(-1, 1))

    return {"prediction": predicted_value[0][0].tolist()}
