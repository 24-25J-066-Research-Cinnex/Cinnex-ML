# Cinnex

Cinnex is a Reaserch project of Final year that have 4 main function base components 
Function 01 -predicts cinnamon prices based on the historical market prices, regional demand and supply data, and economic indicators Gather location-specific data.
Function 02-predict the diseases of cinnamon Leaf based on the symptoms.

## Table of Contents
1. Functions
    -  Cinnamon Price Forecast
        -  Model 1
    -  Leaf Spot Disease
        -  Model 1
2. API
3. How to Setup
4. Others

---

## 1. Functions

### Function 1: Cinnamon Price Forecast
#### Model 1: Linear Regression

- **Dataset (Drive or GitHub URL)**: [Cinnamon Prices Dataset](https://drive.google.com/drive/folders/1bqkIzgF1zOCagfuHsg3Qrw9N_gkPDHck?usp=sharing)
- **Final Code (Folder URL)**: [Source Code](https://github.com/username/repo/tree/main/src)
- **Use Technologies and Model**: Python, scikit-learn, Pandas
- **Model Label**: Cinnamon Price
- **Model Features**: Location, Year, Month, Grade
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

### Function 2: Leaf Spot disease detection
#### Model 2: Leaf Spot disease Idendifier

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
    git clone https://github.com/24-25J-066-Research-Cinnex/Cinnex-ML.git
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
