from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import traceback

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Input schema
class TrainFeatures(BaseModel):
    distance: float
    weather: str
    day_of_week: str
    time_of_day: str
    train_type: str
    route_congestion: str


    
# Home route - serve frontend
@app.get("/")
def home():
    with open("templates/TransitIQ_dashboard.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# Prediction endpoint
@app.post("/predict")
def predict(features: TrainFeatures):
    try:
        input_data = {
            "distance": features.distance,
            "weather": features.weather.strip().title(),
            "day": features.day_of_week.strip().title(),
            "time": features.time_of_day.strip().title(),
            "train": features.train_type.strip().title(),
            "congestion": features.route_congestion.strip().title()
        }

        df = pd.DataFrame([input_data])
        print("INPUT:", df)

        # 🚀 NO encoding here
        prediction = model.predict(df)

        return {"predicted_delay_minutes": round(float(prediction[0]), 2)}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}