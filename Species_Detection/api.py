from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
import imghdr  # For validating image files
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops

# Load the trained model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize the Flask app
app = Flask(__name__)

# Feature extraction function
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 2)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)

    # Use Laplacian edge detection
    laplacian = cv2.Laplacian(enhanced_image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Use Sobel edge detection
    sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

    # Combine Laplacian and Sobel
    combined_edges = cv2.bitwise_or(laplacian, sobel_magnitude)

    # Extract GLCM features from combined edges
    glcm = graycomatrix(combined_edges, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Other statistical features from combined edges
    mean_val = np.mean(combined_edges)
    std_val = np.std(combined_edges)
    entropy_val = -np.sum(combined_edges * np.log2(combined_edges + 1e-5))
    rms_val = np.sqrt(np.mean(combined_edges**2))
    variance = np.var(combined_edges)
    smoothness = 1 - (1 / (1 + np.sum(combined_edges)))
    kurtosis_val = kurtosis(combined_edges.flatten())
    skewness_val = skew(combined_edges.flatten())

    # Correct IDM (Inverse Difference Moment)
    m, n = combined_edges.shape
    row_indices, col_indices = np.indices((m, n))
    IDM = np.sum(combined_edges / (1 + (row_indices - col_indices)**2))
    IDM /= m * n  # Normalize by number of pixels

    # Return all 26 features
    return [contrast, correlation, energy, homogeneity, mean_val, std_val, entropy_val, rms_val, 
            variance, smoothness, kurtosis_val, skewness_val, IDM, laplacian, sobel_magnitude]

# Calculate green percentage function
def calculate_green_percentage(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([35, 50, 50])  # Adjusted lower bound for better green detection
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Perform morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Calculate the percentage of green pixels
    green_pixels = np.sum(green_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    return green_percentage

@app.route("/classify-leaf", methods=["POST"])
def classify_leaf():
    # Check if an image file is uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Please upload an image."}), 400
    
    file = request.files["file"]
    
    # Validate that the file is an image using imghdr
    if not imghdr.what(file):
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    # Read the uploaded image
    np_arr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image file."}), 400

    # Resize the image
    image_resized = cv2.resize(image, (256, 256))

    # Extract features
    features = extract_features(image_resized)
    features_scaled = scaler.transform([features[:-2]])  # Scale the features

    # Predict leaf type
    prediction = model.predict(features_scaled)
    if prediction[0] == 1:
        leaf_type = "Pure Cinnamon"
        green_percentage = calculate_green_percentage(image_resized)
        health_status = "Healthy" if green_percentage > 60 else "Unhealthy"
        return jsonify({
            "leaf_type": leaf_type,
            "health_status": health_status
        })
    else:
        return jsonify({"leaf_type": "Wild Cinnamon"})


# Run the app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
