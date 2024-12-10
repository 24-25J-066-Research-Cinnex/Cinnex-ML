from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pickle
import logging
import pandas as pd


app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load data during startup
df = pd.read_csv('data\Cinnamon_GradePrices_LKR.csv') 
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

class PredictionRequest(BaseModel):
    location: str
    grade: str
    forecast_date: str

class NewPredictionRequest(BaseModel):
    location: str
    grade: str
    forecast_date: str

def LoadModel():
    filename = 'models\cinnamon-price-forecast-LKR.pickle'
    #filename = "models/arima_model.pickel"
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {filename}")
        raise HTTPException(status_code=500, detail="Mode file not found")
    except Exception as e:
        logging.error(f"Error during model loading: {e}")
        raise HTTPException(status_code=500, detail="Model loading error")


def prediction(lst):

    try:
        model = LoadModel()
        pred = model.predict([lst])
        return pred.tolist()  # Ensure the result is JSON-serializable

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the CinneX application"}

@app.post("/predict")
async def index(request: NewPredictionRequest):
    try:
        location= request.location
        grade = request.grade
        forecast_date= request.forecast_date
       
        logging.info(f"Received request: {location},{grade}, {forecast_date}")
        #Received request: Galle,Alba, 2025-05-30

        feature_list = [None]*5
        feature_list[0] = str(location)

        location_list = ['Galle', 'Matara', 'Kalutara']
        grade_list = ['Alba (Average Price)', 'C-5 Sp (Average Price)', 'C-5 (Average Price)', 'C-4 (Average Price)', 'H-2 (Average Price)']
        
        #travel through the list to find the index
        def traverse(lst, value):
            for index, item in enumerate(lst):
                if item == value:
                    return index
            return -1

        Loacation_index = traverse(location_list, location)
        grade_index = traverse(grade_list, grade)

        if Loacation_index == -1 or grade_index == -1 :
            raise HTTPException(status_code=400, detail="Invalid location or grade")

        feature_list[1] = grade_index
        feature_list[2] = datetime.strptime(forecast_date, '%Y-%m-%d').strftime('%Y-%m-%d')  # Convert to datetime object

        pred = prediction(feature_list)
        return {"newprediction": pred}
    except HTTPException as http_exc:
        logging.error(f"HTTP error in /predict endpoint: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")