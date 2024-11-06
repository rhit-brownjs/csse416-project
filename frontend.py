from flask import Flask, render_template, request
import numpy as np
# Import your trained model here. For example:
# from your_model import predict_price

app = Flask(__name__)

# List of available house images
house_images = [f'house{i}.jpg' for i in [4,38,44,48,89,227]]  # Adjust range based on the number of images

@app.route('/')
def index():
    return render_template('index.html', images=house_images)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        features = [float(request.form['feature1']), float(request.form['feature2']), float(request.form['feature3'])]  # Update as needed
        selected_image = request.form['house_image']  # Get selected image name
        
        # Process the selected image with features if needed by your model
        # Convert features to numpy array and reshape for model input
        features_array = np.array(features).reshape(1, -1)
        # Call your model's prediction function
        prediction = predict_price(features_array, selected_image)  # Adjust as needed
        
        return render_template('predict.html', prediction=prediction, selected_image=selected_image)

if __name__ == '__main__':
    app.run(debug=True)
