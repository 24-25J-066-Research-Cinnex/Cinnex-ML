from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and label encoder
try:
    model = joblib.load('multi_svm_model.pkl')  # Load the trained Multi-SVM model
    label_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Convert uploaded file to an OpenCV image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400

        # Preprocess image
        print("[API] Image shape after decoding:", image.shape)

        # Resize image to 256x256
        image_resized = cv2.resize(image, (256, 256))
        print("[API] Image shape after resizing:", image_resized.shape)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        print("[API] Image shape after converting to grayscale:", gray_image.shape)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        print("[API] Blurred image mean value:", np.mean(blurred_image))

        # Sobel Edge Detection (instead of Canny)
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

        print("[API] Sobel magnitude (First 10 values):", sobel_magnitude.flatten()[:10])

        # Normalize Sobel edges
        sobel_magnitude_normalized = sobel_magnitude.astype('float32') / 255.0
        print("[API] Normalized Sobel Edges (First 10 values):", sobel_magnitude_normalized.flatten()[:10])

        # Flatten the edge data
        image_data = sobel_magnitude_normalized.flatten()

        # Print the final image data shape and first 10 values
        print("[API] Image data shape:", image_data.shape)
        print("[API] First 10 values:", image_data[:10])

        # Predict the grade
        prediction = model.predict([image_data])
        predicted_label = label_encoder.inverse_transform(prediction)

        return jsonify({'predicted_grade': predicted_label[0]}), 200
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
