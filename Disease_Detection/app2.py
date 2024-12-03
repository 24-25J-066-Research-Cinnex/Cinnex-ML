import uuid
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import pickle
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import cv2

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the saved model and label binarizer
try:
    model = tf.keras.models.load_model('./models/Cinnamon_LeafSpot_disease_model.keras')
    with open('./models/Cinnamon_LeafSpot_disease_model_label_binarizer.pkl', 'rb') as f:
        label_binarizer = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model or label binarizer: {e}")
    raise

# Assuming the model expects an input shape of (128, 128, 3)
DEFAULT_IMAGE_SIZE = (128, 128)

def preprocess_image(file: UploadFile):
    try:
        # Read the file content and convert it to an io.BytesIO object
        image_data = file.file.read()
        image = tf.keras.preprocessing.image.load_img(BytesIO(image_data), target_size=DEFAULT_IMAGE_SIZE)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.array([image_array])  # Convert single image to a batch.
        logging.info(f"Preprocessed image shape: {image_array.shape}")
        return image_array
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Image preprocessing error")

def predict_disease(image_array):
    try:
        prediction = model.predict(image_array)
        logging.info(f"Prediction raw output: {prediction}")
        predicted_class_label = label_binarizer.inverse_transform(prediction)
        logging.info(f"Predicted class label: {predicted_class_label}")
            
        # Ensure the predicted class label is JSON-serializable
        return str(predicted_class_label[0])
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error.")

def calculate_disease_spread(image_data, predicted_class_label):
    try:
        if predicted_class_label != 'Cinnamon_Healthy Leaf':  # Assuming 'Cinnamon_Healthy Leaf' is the healthy class
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define color range for diseased areas (adjust as needed)
            lower_range = np.array([0, 50, 50])  # Example lower range for brown/yellowish diseased areas
            upper_range = np.array([30, 255, 255]) # Example upper range for brown/yellowish diseased areas

            # Create a mask for diseased areas
            mask = cv2.inRange(img_hsv, lower_range, upper_range)

            # Calculate disease spread percentage
            diseased_pixels = cv2.countNonZero(mask)
            total_pixels = img.shape[0] * img.shape[1]
            disease_spread_percentage = (diseased_pixels / total_pixels) * 100
            logging.info(f"disease_spread_percentage: {disease_spread_percentage}")

            return str(disease_spread_percentage)
        else:
            return 0  # Healthy leaf, 0% disease spread
    except Exception as e:
        logging.error(f"Error during disease spread calculation: {e}")
        raise HTTPException(status_code=500, detail="Disease spread calculation error")

@app.post("/newpredict")
async def predict(file: UploadFile):
    request_id = uuid.uuid4()
    logging.info(f"Received file: {file.filename}, Request ID: {request_id}")
    try:
        # Read the file content
        image_data = file.file.read()

        # Preprocess the image
        image_array = preprocess_image(file)
        logging.info(f"Image array for prediction: {image_array}")

        # Predict the disease
        predicted_class_label = predict_disease(image_array)

        # Calculate disease spread percentage
        disease_spread_percentage = calculate_disease_spread(image_data, predicted_class_label)
        logging.info(f"Disease spread percentage: {disease_spread_percentage}%")

    except HTTPException as http_exc:
        logging.error(f"HTTP error in /newpredict endpoint, Request ID: {request_id}, Detail: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Unexpected error in /newpredict endpoint, Request ID: {request_id}, Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

    # Wrap the response in a dict
    return {
        "prediction": predicted_class_label,
        "disease_spread_percentage": disease_spread_percentage,
        "request_id": str(request_id)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)