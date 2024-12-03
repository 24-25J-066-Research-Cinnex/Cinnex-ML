from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

class PredictionRequest(BaseModel):
    location: str
    year: int
    month: str
    quantity: float

class NewPredictionRequest(BaseModel):
    year: int
    month: str
    grade: str
    region: str

def LoadModel():
    filename = "models/predictor.pickle"
    #filename = "models/predictor1.pickle"
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

def LoadNewModel():
    filename = "models/predictor1.pickle"
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
    
def newprediction(lst):

    try:
        model = LoadNewModel()
        pred = model.predict([lst])
        return pred.tolist()  # Ensure the result is JSON-serializable

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")
    
    

@app.get("/")
async def read_root():
    return {"message": "Welcome to the CinneX application"}


@app.post("/predict")
async def index(request: PredictionRequest):
    try:
        location = request.location
        year = request.year
        month = request.month
        quantity = request.quantity

        logging.info(f"Received request: {location}, {year}, {month}, {quantity}")

        feature_list = [0] * 24  # Initialize a list with 24 features
        feature_list[0] = int(year)
        feature_list[1] = float(quantity)

        location_list = ['anuradhapura', 'badulla', 'colombo', 'galle', 'hambantota', 'kandy', 'kurunegala', 'matale', 'ratnapura']
        month_list = ['april', 'august', 'december', 'february', 'january', 'july', 'june', 'march', 'may', 'november', 'october', 'september']

        def traverse(lst, value):
            for index, item in enumerate(lst):
                if item == value:
                    return index
            return -1

        location_index = traverse(location_list, location)
        month_index = traverse(month_list, month)

        if location_index == -1 or month_index == -1:
            raise HTTPException(status_code=400, detail="Invalid location or month")

        feature_list[2] = location_index
        feature_list[3] = month_index

        pred = prediction(feature_list)
        return {"prediction": pred}
    except HTTPException as http_exc:
        logging.error(f"HTTP error in /predict endpoint: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post("/newpredict")
async def index(request: NewPredictionRequest):
    try:
        year = request.year
        month = request.month
        grade = request.grade
        region = request.region

        logging.info(f"Received request: {year}, {month}, {grade}, {region}")
        #Received request: 2025, 2, C5, Galle

        feature_list = [0] * 61  # Initialize a list with 62 features
        feature_list[0] = int(year)

        region_list = ['Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo', 'Galle', 'Gampaha', 'Hambantota',
           'Jaffna', 'Kalutara', 'Kandy', 'Kegalle', 'Kilinochchi', 'Kurunegala', 'Mannar', 'Matale',
           'Matara', 'Monaragala', 'Mullaitivu', 'Nuwara Eliya', 'Polonnaruwa', 'Puttalam', 'Ratnapura',
           'Trincomalee', 'Vavuniya']
        month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
          'October', 'November', 'December']
        grade_list = ['C1', 'C2', 'C3', 'C4', 'C5']
        def traverse(lst, value):
            for index, item in enumerate(lst):
                if item == value:
                    return index
            return -1

        region_index = traverse(region_list, region)
        month_index = traverse(month_list, month)
        grade_index = traverse(grade_list, grade)

        if region_index == -1 or month_index == -1 or grade_index == -1:
            raise HTTPException(status_code=400, detail="Invalid location or month")

        feature_list[1] = month_index
        feature_list[2] = grade_index
        feature_list[3] = region_index

        pred = newprediction(feature_list)
        return {"newprediction": pred}
    except HTTPException as http_exc:
        logging.error(f"HTTP error in /predict endpoint: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")