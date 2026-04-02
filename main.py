from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import traceback

app = FastAPI(title="Weather Classification API")

with open("weather_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

try:
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except:
    le = None


class WeatherInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Precipitation: float
    Atmospheric_Pressure: float
    UV_Index: int
    Visibility_km: float
    Cloud_Cover: str
    Season: str
    Location: str


@app.get("/")
def home():
    return {"message": "Weather Classification API is running 🚀"}


@app.post("/predict")
def predict(data: WeatherInput):
    try:
      
        df = pd.DataFrame([{
            "Temperature": data.Temperature,
            "Humidity": data.Humidity,
            "Wind Speed": data.Wind_Speed,
            "Precipitation (%)": data.Precipitation,
            "Atmospheric Pressure": data.Atmospheric_Pressure,
            "UV Index": data.UV_Index,
            "Visibility (km)": data.Visibility_km,
            "Cloud Cover": data.Cloud_Cover,
            "Season": data.Season,
            "Location": data.Location
        }])

        df["Cloud Cover"] = df["Cloud Cover"].str.lower().str.strip()
        df["Season"] = df["Season"].str.lower().str.strip()
        df["Location"] = df["Location"].str.lower().str.strip()

        df["temp_humidity"] = df["Temperature"] * df["Humidity"]
        df["wind_pressure"] = df["Wind Speed"] * df["Atmospheric Pressure"]

        print("Final Columns:", df.columns)

        prediction = model.predict(df)

        prediction_value = int(prediction[0])

        if le is not None and hasattr(le, "classes_"):
            if prediction_value < len(le.classes_):
                prediction_label = le.classes_[prediction_value]
            else:
                prediction_label = f"Unknown ({prediction_value})"
        else:
            prediction_label = prediction_value

        prediction_label = str(prediction_label)

        return {"prediction": prediction_label}

    except Exception as e:
        print("ERROR TRACE:")
        print(traceback.format_exc())
        return {"error": str(e)}