from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pickle
import cv2
from tensorflow import keras
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the saved model and label binarizer
model = keras.models.load_model('./models/Disease_classification_model.keras')
#model = keras.models.load_model('./models/Cinnamon_disease_model.keras')
with open('./models/cinnamon_disease_label_transform.pkl', 'rb') as f:
    label_binarizer = pickle.load(f)

DEFAULT_IMAGE_SIZE = (256, 256)

def preprocess_image(file: UploadFile):
    try:
        image = Image.open(BytesIO(file.file.read())).convert("RGB")
        image = image.resize(DEFAULT_IMAGE_SIZE)
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

def predict_disease(image_array):
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = label_binarizer.classes_[1 - predicted_class_index]  # Swap index for troubleshooting
    return predicted_class_label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Preprocess and predict
    try:
        image_array = preprocess_image(file)
        predicted_class_label = predict_disease(image_array)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"prediction": predicted_class_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)