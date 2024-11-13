from typing import OrderedDict
from flask import Flask, render_template, request
import numpy as np

import xgboost as xgb
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import VGG16_Weights

app = Flask(__name__)

# List of available house images
house_images = [f'{i}.jpg' for i in [4,38,44,48,89,227]]  # Adjust range based on the number of images

# Load the CSV file again
data_updated = pd.read_csv('data/socal.csv')

# Create a dictionary mapping from 'citi' to 'n_citi'
city_mapping = dict(zip(data_updated['citi'], data_updated['n_citi']))
city_mapping = OrderedDict(sorted(city_mapping.items()))



def predict_price(features, img):

    # Function to extract features from the image
    def extract_features(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros(4096)  # Return a zero vector if image is not found or corrupted
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image)
        return features.numpy().flatten().reshape(1, -1)

    # Prepare the text features
    def preprocess_text_features(features):
        return np.array(features).reshape(1, -1)  # Ensure it is a 2D 
    
    # Load the pre-trained XGBoost model
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("xgb_model.json")

    # Load the pretrained VGG16 model. Setting `pretrained=True` loads weights trained on ImageNet.
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    # We only need the features, so we remove the classifier part by taking only `model.features`.
    model = model.features
    # Set the model to evaluation mode to prevent training-related behavior, such as dropout.
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG16 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # VGG16 normalization
                            std=[0.229, 0.224, 0.225])
    ])

    # Combine image and text features
    
    image_features = extract_features("data/socal_pics/" + img)
    text_features = preprocess_text_features(features)

    # Concatenate all features
    all_features = np.hstack((text_features, image_features))

    # Predict with the XGBoost model
    predicted_price = xgb_model.predict(all_features)

    return predicted_price[0]

@app.route('/')
def index():
    return render_template('index.html', images=house_images, city_mapping=city_mapping)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        features = [int(request.form['feature1']), int(request.form['feature2']), int(request.form['feature3']), int(request.form['feature4'])]  # Update as needed
        selected_image = request.form['house_image']  # Get selected image name
        
        # Process the selected image with features if needed by your model
        # Convert features to numpy array and reshape for model input
        features_array = np.array(features).reshape(1, -1)
        # Call your model's prediction function
        prediction = predict_price(features_array, selected_image)  # Adjust as needed
        
        return render_template('predict.html', prediction=prediction, selected_image=selected_image)

if __name__ == '__main__':
    app.run(debug=True)
