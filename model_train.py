import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("train.csv")

# ✅ Rename columns (CLEAN + SIMPLE)
df.columns = [
    "distance",
    "weather",
    "day",
    "time",
    "train",
    "delay",
    "congestion"
]

print("COLUMNS:", df.columns)

# Features & target
X = df.drop("delay", axis=1)
y = df["delay"]

# Columns
cat_cols = ["weather", "day", "time", "train", "congestion"]
num_cols = ["distance"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")