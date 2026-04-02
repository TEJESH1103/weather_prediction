#  Weather Classification API

##  Project Overview
This project predicts the **weather type** (Sunny, Rainy, Cloudy, Snowy) based on environmental conditions using a Machine Learning model deployed with FastAPI.

---

##  Features
- Predicts weather type using ML model
- Uses Random Forest with optimized parameters
- Includes feature engineering
- Deployed using FastAPI
- Returns human-readable predictions

---

## Machine Learning Pipeline
- Data preprocessing (scaling + encoding)
- Feature engineering:
  - temp_humidity
  - wind_pressure
- Feature selection (SelectKBest)
- Model: Random Forest Classifier

---

##  Project Structure
weather-project/
│
├── app.py
├── weather_pipeline.pkl
├── label_encoder.pkl
├── requirements.txt
├── train.ipynb
└── README.md

---

##  Installation

pip install -r requirements.txt

---

##  Run the API

uvicorn main:app --reload

---

##  API Endpoint

### POST `/predict`

### Sample Input
{
  "Temperature": 30,
  "Humidity": 70,
  "Wind_Speed": 12,
  "Precipitation": 20,
  "Atmospheric_Pressure": 1012,
  "UV_Index": 7,
  "Visibility_km": 8,
  "Cloud_Cover": "partly cloudy",
  "Season": "summer",
  "Location": "inland"
}

---

### Sample Output
{
  "prediction": "sunny"
}

---

## Model Performance
- Accuracy: ~91%
- F1 Score: ~0.91

---

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- Pandas & NumPy

---

## Future Improvements
- Add prediction confidence
- Deploy on cloud (Render)
- Build frontend UI

---

##  Author
Mamidipaka Venkata Sai Tejesh
