import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Path for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('fashion_mnist_cnn.h5')

# Class names for Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads and classification
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(28, 28), color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions[0])
        confidence = predictions[0][predicted_label]

        # Return prediction result
        return jsonify({
            "category": class_names[predicted_label],
            "confidence": float(confidence),
            "file_path": filepath
        })

# Run the Flask app
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
