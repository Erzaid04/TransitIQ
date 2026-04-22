from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('feature_columns.pkl','rb') as f:
    feature_columns = pickle.load(f)
    
@app.get("/")
def home():
    return {"Message":"TransitIQ API is running"}

class TrainFeatures(BaseModel):
    distance: float
    weather: str
    day_of_week: str
    train_type: str
    time_of_day: str
    route_congestion: str
    
@app.post("/predict")
def predict(features: TrainFeatures):
    try:
        input_data = {
            'Distance_Between_Stations_km': features.distance,
            'Weather_Conditions': features.weather,
            'Day_of_Week': features.day_of_week,
            'Train_type': features.train_type,
            'Time_of_Day': features.time_of_day,
            'Route_Congestion': features.route_congestion
        }
        
        df_input = pd.DataFrame([input_data])
        print("Input DF: ",df_input)
        df_encoded = pd.get_dummies(df_input)
        print("Encoded DF columns: ",df_encoded.columns.tolist())
        df_encoded = df_encoded.reindex(columns=feature_columns,fill_value=0)
        print("Reindexed shape: ",df_encoded.shape)
        prediction = model.predict(df_encoded)
        return {"Predicted_delay_minutes":round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e)}