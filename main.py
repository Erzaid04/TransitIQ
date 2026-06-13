from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import traceback
import sqlite3
from datetime import datetime

app = FastAPI()

# ---------------------- DATABASE SETUP ----------------------

def init_db():
    conn = sqlite3.connect("transitiq.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            distance REAL,
            weather TEXT,
            day_of_week TEXT,
            time_of_day TEXT,
            train_type TEXT,
            route_congestion TEXT,
            predicted_delay_minutes REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

# Call DB init at startup
init_db()

def save_prediction(features, prediction):
    conn = sqlite3.connect("transitiq.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_history 
        VALUES (NULL,?,?,?,?,?,?,?,?)
    """, (
        features.distance,
        features.weather,
        features.day_of_week,
        features.time_of_day,
        features.train_type,
        features.route_congestion,
        prediction,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

# ---------------------- CORS ----------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- LOAD MODEL ----------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---------------------- INPUT SCHEMA ----------------------

class TrainFeatures(BaseModel):
    distance: float
    weather: str
    day_of_week: str
    time_of_day: str
    train_type: str
    route_congestion: str

# ---------------------- ROUTES ----------------------

# Home route
@app.get("/")
def home():
    with open("templates/TransitIQ_dashboard.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# Prediction route
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

        print("FINAL DF:", df)  # debug

        prediction = model.predict(df)
        result = round(float(prediction[0]), 2)

        save_prediction(features, result)

        return {"predicted_delay_minutes": result}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
# History route
@app.get("/history")
def get_history():
    conn = sqlite3.connect("transitiq.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM prediction_history 
        ORDER BY id DESC 
        LIMIT 10
    """)

    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "distance": row[1],
            "weather": row[2],
            "day_of_week": row[3],
            "time_of_day": row[4],
            "train_type": row[5],
            "route_congestion": row[6],
            "predicted_delay": row[7],
            "created_at": row[8]
        })

    return {"history": history}