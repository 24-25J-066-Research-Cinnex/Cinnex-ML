from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import joblib
import logging
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load data and models during startup
DATA_PATH = 'data\Cinnamon_GradePrices_LKR.csv'
MODEL_PATH = 'models\cinnamon-price-forecast-LKR.pickle'

try:
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    logging.info(f"Data loaded successfully from {DATA_PATH}")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise RuntimeError("Failed to load data file")

try:
    arima_model = joblib.load(MODEL_PATH)
    logging.info(f"ARIMA model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading ARIMA model: {e}")
    raise RuntimeError("Failed to load ARIMA model")

# Input validation
class PredictionRequest(BaseModel):
    location: str = Field(..., example="Galle")
    grade: str = Field(..., example="Alba")
    forecast_date: str = Field(..., example="2025-05-30")

# Helper functions
def prepare_time_series(df, location, grade):
    filtered_df = df[df['District'] == location]
    if grade not in filtered_df.columns:
        raise ValueError(f"Grade '{grade}' not found in data")
    return filtered_df[grade]

def forecast_price(series, forecast_date):
    try:
        # Fit ARIMA model to the series
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()

        # Calculate days into the future
        last_date = series.index[-1]
        days_ahead = (pd.to_datetime(forecast_date) - last_date).days

        if days_ahead <= 0:
            raise ValueError("Forecast date must be in the future")

        forecast_result = model_fit.forecast(steps=days_ahead)
        return forecast_result[-1]  # Return the price for the forecasted date
    except Exception as e:
        logging.error(f"Error during forecasting: {e}")
        raise ValueError("Forecasting failed")

# API Endpoints
@app.get("/")
async def read_root():
    return {"message": "Welcome to the CinneX application"}

@app.post("/predict")
async def predict_price(request: PredictionRequest):
    try:
        # Parse inputs
        location = request.location
        grade = request.grade + ' '+'(Average Price)'
        forecast_date = request.forecast_date

        # Validate date
        try:
            forecast_date_parsed = pd.to_datetime(forecast_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid forecast_date format. Use YYYY-MM-DD")

        # Prepare data
        series = prepare_time_series(df, location, grade)

        # Make prediction
        predicted_price = forecast_price(series, forecast_date_parsed)

        logging.info(f"Prediction successful: {predicted_price}")
        return {"location": location, "grade": grade, "forecast_date": forecast_date, "predicted_price": predicted_price}
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
