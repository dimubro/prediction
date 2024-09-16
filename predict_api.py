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


model_path = 'E:/Final-project/ml/age_sex_model.pkl'


print("Loading the model...")
with open(model_path, 'rb') as file:
    model = pickle.load(file)
print("Model loaded successfully!")


def load_image_from_url(url, target_size=(48, 48)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize(target_size)  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.get_json()
        image_url = data.get('image_url')

       
        input_image = load_image_from_url(image_url)

        
        prediction = model.predict(input_image)

        
        predicted_gender = prediction[0][0]  
        predicted_age = prediction[1][0]   
        
       
        gender = 'Male' if predicted_gender < 0.5 else 'Female'

        
        return jsonify({
            'age': int(np.round(predicted_age)),
            'gender': gender
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)