# import tensorflow as tf
# from tensorflow import keras
# import pickle

# model_path = 'E:/Final-project/ml/age_sex_model.pkl'

# print("Loading the model...")  # Print message before loading the model

# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# print("Model loaded successfully!")  # Print message after loading the model

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO

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

# Example image URL (replace with your own)
image_url = 'https://scontent.fcmb11-1.fna.fbcdn.net/v/t1.6435-9/97979822_2685105178413607_4696806467733291008_n.jpg?_nc_cat=104&ccb=1-7&_nc_sid=53a332&_nc_eui2=AeGmGzSElXXqXPLsO1XiuRcsogbhPrsr9qmiBuE-uyv2qaggeLIXKdYcKJg4t_AVmtFY1A6uiLGF4C5kQPr9f2ki&_nc_ohc=jOPmXXVp50IQ7kNvgGjk3rU&_nc_ht=scontent.fcmb11-1.fna&_nc_gid=AIDeAKyuB1U8crdGJZiwaPy&oh=00_AYABDsJysfkRTb8OooDKNAGCz0Bb2aD7oOW6Fl4aqR5u6w&oe=670D58E9'
# Load and preprocess the image
print("Loading and preprocessing image...")
input_image = load_image_from_url(image_url)

# Make a prediction
print("Making prediction...")
prediction = model.predict(input_image)

# Extract predicted gender and age from the model's output
# Assuming prediction[0] is gender and prediction[1] is age
predicted_gender = prediction[0][0]  # Adjust index based on your model output
predicted_age = prediction[1][0]     # Adjust index based on your model output

# Determine gender from the prediction
gender = 'Male' if predicted_gender < 0.5 else 'Female'  # Modify threshold based on model

# Print the results
print(f"Predicted Age: {int(np.round(predicted_age))}")
print(f"Predicted Gender: {gender}")