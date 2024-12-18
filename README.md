# csse416-project

## Overview
Our projecis for predicting the price of houses in Southern California.

The Minimum notebook predicts the price of a house based on square footage, city, number of beds, and number of baths. No image data is used for the minimum. This prediction beats the simple bias regressor.

The Reasonable notebook predicts the price of a house based on square footage, city, number of beds, number of baths and a house image. The image features are extracted using VGG16 and then concatenated with the textual features. XGBoost is used for regression to create predictions given the features. The RMSE beats that of our Minimum notebook which uses only textual features. 

The Stretch notebook uses the saved model weights to load the XGBoost model and create a prediction. This is the basis of our flask website. 

All notebooks can be run using the same method used to run all Jupyter Notebooks, simply hit the "Run All" button.

Note: Before running the web application, ensure all external packages are installed. 

This can be done efficiently by using the following command: "pip install -r requirements.txt" in the project's root directory

For the web application, using Visual Studio code, you can run frontend.py using the "Run" button. The web app will be locally hosted on your machine at port 5000. The url given to you in the terminal should look like this: http://127.0.0.1:5000/

## Packages
Flask

numpy

xgboost

pandas

torch

torchvision

scikit-learn

Pillow

## Data
The dataset came from Kaggle: https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal

In our project, the data is stored in the data folder. The data folder stores the socal.csv and socal_pics, which holds all the pictures of the houses.

For adding a picture to the web application, a picture from socal_pics must also be copied into static/images. Then, the picture name/number can be added to the house_images list in the frontend.py file.
