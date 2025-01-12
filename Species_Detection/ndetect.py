import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt




# Function to extract features from an image using both Laplacian and Sobel Edge Detection

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

# Step 1: Load the saved model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Step 2: Open file dialog to select an image
root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(title="Select an Image")

# Check if a file was selected
if not file_path:
    print("No file selected. Exiting...")
    exit(1)

# Load the selected image
image = cv2.imread(file_path)
image_resized = cv2.resize(image, (256, 256))

# Step 3: Extract features from the image
features = extract_features(image_resized)

# Step 4: Scale the features using the saved scaler
features_scaled = scaler.transform([features[:-2]])  # Use all features except the last two for scaling (laplacian, sobel)

# Step 5: Predict the leaf type using the trained model
prediction = model.predict(features_scaled)
print(f"Predicted Leaf Type: {'Pure Cinnamon' if prediction[0] == 1 else 'Wild Cinnamon'}")

# Step 6: Display the images in the same window
laplacian_image = features[-2]  # Extracted Laplacian image
sobel_image = features[-1]      # Extracted Sobel image

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original Image
axes[0].imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis('off')

# Laplacian Image
axes[1].imshow(laplacian_image, cmap='gray')
axes[1].set_title("Laplacian Edge Detection")
axes[1].axis('off')

# Sobel Image
axes[2].imshow(sobel_image, cmap='gray')
axes[2].set_title("Sobel Edge Detection")
axes[2].axis('off')

# Show the images
plt.show()
