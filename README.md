# Cinnex

Cinnex is a machine learning project that predicts cinnamon prices based on the historical market prices, regional demand and supply data, and economic indicators Gather location-specific data. Other hand, it also predict the diseases of cinnamon plants based on the symptoms and environmental conditions.

## Table of Contents
1. Functions
    -  Function 1
        -  Model 1
    -  Function 2
        -  Model 1
2. API
3. How to Setup
4. Others

---

## 1. Functions

### Function 1: Predict cinnamon Prices
#### Model 1: Linear Regression

- **Dataset (Drive or GitHub URL)**: [Cinnamon Prices Dataset](https://drive.google.com/drive/folders/1bqkIzgF1zOCagfuHsg3Qrw9N_gkPDHck?usp=sharing)
- **Final Code (Folder URL)**: [Source Code](https://github.com/username/repo/tree/main/src)
- **Use Technologies and Model**: Python, scikit-learn, Pandas
- **Model Label**: Price
- **Model Features**: Location, Year, Month, Quantity
<!-- - **Model (GitHub or Drive URL)**: [Model File](https://github.com/username/repo/blob/main/models/linear_regression.pkl) -->
<!-- - **Tokenizer (GitHub or Drive URL)**: [Tokenizer File](https://github.com/username/repo/blob/main/tokenizers/tokenizer.pkl) -->
<!-- - **Scaler (GitHub or Drive URL)**: [Scaler File](https://github.com/username/repo/blob/main/scalers/scaler.pkl) -->
- **Accuracy**:0.91
<!-- - **How to Load and Get Prediction for One Input**:
    ```python
    import joblib

    model = joblib.load('models/linear_regression.pkl')
    scaler = joblib.load('scalers/scaler.pkl')

    input_data = [[2500, 4, 'Suburban']]
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    print("Predicted Price:", prediction)
    ``` -->

### Function 2: Spot disease detection
#### Model 2: Spot disease classifier

- **Dataset (Drive or GitHub URL)**: [Cinnamon Plant Stem and Branch Disease Dataset](https://www.kaggle.com/datasets/madhavipethangoda/cinnamon-plant-stem-and-branch-disease-dataset)
- **Final Code (Folder URL)**: [Source Code](https://github.com/username/repo/tree/main/src)
- **Use Technologies and Model**: Python, scikit-learn, TensorFlow, Keras
- **Model Label**: Disease
- **Model Features**: 
- **Model (GitHub or Drive URL)**: [Models](https://drive.google.com/drive/folders/13laSrkmOlaAqsgmI9yn4KgwKa0190c2H?usp=sharing)
- **Accuracy**:0.95
- **How to Load and Get Prediction for One Input**:
    ```python
    import numpy as np
    import pickle
    import cv2
    import matplotlib.pyplot as plt
    from tensorflow import keras

    model = keras.models.load_model('/content/drive/MyDrive/Cinnex/Model/Disease_classification_model.keras')
    with open('/content/drive/MyDrive/Cinnex/Model/cinnamon_disease_label_transform.pkl', 'rb') as f:
        label_binarizer = pickle.load(f)

    DEFAULT_IMAGE_SIZE = tuple((256, 256))

    def convert_image_to_array(image_dir):
        try:
            image = cv2.imread(image_dir)
            if image is not None :
                image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
                return img_to_array(image)
            else :
                return np.array([])
        except Exception as e:
            print(f"Error : {e}")
            return None

    def predict_disease(image_path):
        image_array = convert_image_to_array(image_path)
        np_image = np.array(image_array, dtype=np.float16) / 255.0
        np_image = np.expand_dims(np_image,0)

        prediction = model.predict(np_image)

        predicted_class_index = np.argmax(prediction)
        predicted_class_label = label_binarizer.classes_[1 - predicted_class_index]

        plt.imshow(plt.imread(image_path))
        plt.title(f"Prediction: {predicted_class_label}")
        plt.axis('off')
        plt.show()

        return predicted_class_label

    # Example usage:
    predicted_label = predict_disease('@image_path')
    print(f"Predicted Disease: {predicted_label}")
    ```
---

## 2. API

<!-- - **Use Technology**: FastAPI
- **API Folder (Drive or GitHub URL)**: [API Source Code](https://github.com/username/repo/tree/main/api)
- **API Folder Screenshot**: 
    - ![API Folder Screenshot](https://example.com/screenshot.png)
- **API Testing Swagger Screenshots for All Endpoints**:
    - ![Swagger Endpoint 1](https://example.com/swagger1.png) -->

---

## 3. How to Setup

### Pre-requisites
- Python v3.10 or higher
- pip v21.2.4 or higher

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/SilverlineIT/Cinnex-ML.git
    ```
2. Navigate to the project directory:
    ```bash
    cd cinnex-ml
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    fastapi dev Component_1/app.py # For Component 1
    fastapi dev Component_2/app.py # For Component 2
    ```

5. How to activate environment
   ```bash
    python -m venv cinnex
    cinnex\Scripts\activate
    ```

## 4. Others
"# Cinnex-ML" 
