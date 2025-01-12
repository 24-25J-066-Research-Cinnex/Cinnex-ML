import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
import tkinter as tk
from tkinter import filedialog

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
            variance, smoothness, kurtosis_val, skewness_val, IDM]

# Step 1: Open file dialog to select the folder containing images
root = tk.Tk()
root.withdraw()  # Hide the root window
folder_name = filedialog.askdirectory(title="Select Folder Containing Images")

# Check if the folder exists
if not folder_name:
    print("No folder selected. Exiting...")
    exit(1)

# Display the selected folder
print(f"Selected folder: {folder_name}")

# Define file extensions
valid_extensions = ['.png', '.jpg', '.bmp', '.JPG']
all_images = [f for f in os.listdir(folder_name) if any(f.endswith(ext) for ext in valid_extensions)]

# Sort images numerically based on the numerical part of the filename (e.g., img_01, img_02, ..., img_300)
all_images.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

# Initialize dataset list and leaftype list
dataset = []
leaftype = []

# Process images in the correct numerical order
for img_name in all_images:
    image_path = os.path.join(folder_name, img_name)
    print(f"Processing image: {img_name}")

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    features = extract_features(image)
    dataset.append(features)

    # Assign leaf type based on image number (1 for Pure Cinnamon, 2 for Wild Cinnamon)
    img_num = int(img_name.split('_')[1].split('.')[0])
    if img_num <= 150:
        leaftype.append(1)  # Type 1 for Pure Cinnamon
    else:
        leaftype.append(2)  # Type 2 for Wild Cinnamon

# Convert to numpy arrays for easier handling
dataset = np.array(dataset)
leaftype = np.array(leaftype)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, leaftype, test_size=0.3, random_state=42)

# Step 3: Standardize the data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Use GridSearchCV to find the best SVM hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Step 5: Evaluate the model on the test set
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Save the model and scaler
joblib.dump(best_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")
