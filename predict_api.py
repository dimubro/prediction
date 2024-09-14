from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Path to your pickled model
model_path = 'E:/Final-project/ml/age_sex_model.pkl'

# Load the pickled model
print("Loading the model...")
with open(model_path, 'rb') as file:
    model = pickle.load(file)
print("Model loaded successfully!")

# Function to load an image from a URL
def load_image_from_url(url, target_size=(48, 48)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize(target_size)  # Resize to the model's expected input size (48x48)
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image (if required by your model)
    return img_array

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image URL from the POST request (from PHP)
        data = request.get_json()
        image_url = data.get('image_url')

        # Load and preprocess the image
        input_image = load_image_from_url(image_url)

        # Make a prediction
        prediction = model.predict(input_image)

        # Extract predicted gender and age from the model's output
        predicted_gender = prediction[0][0]  # Adjust index based on your model output
        predicted_age = prediction[1][0]     # Adjust index based on your model output

        # Determine gender from the prediction
        gender = 'Male' if predicted_gender < 0.5 else 'Female'

        # Return the response as JSON
        return jsonify({
            'age': int(np.round(predicted_age)),
            'gender': gender
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)