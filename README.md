# csse416-project

## Overview
Our projecis for predicting the price of houses in Southern California.

The Minimum notebook predicts the price of a house based on square footage, city, number of beds, and number of baths. No image data is used for the minimum. This prediction beats the simple bias regressor.
The Reasonable notebook
The Stretch notebook
All notebooks can be run using the same method used to run all Jupyter Notebooks, simply hit the "Run All" button.

For the web application, using Visual Studio code, you can run frontend.py using the "Run" button. The web app will be locally hosted on your machine at port 5000. The url given to you in the terminal should look like this: http://127.0.0.1:5000/

## Packages
flask
numpy
xgboost
pandas
torch
torchvision
scikit-learn

## Data
The dataset came from Kaggle: https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal

In our project, the data is stored in the data folder. The data folder stores the socal.csv and socal_pics, which holds all the pictures of the houses.

For adding a picture to the web application, a picture from socal_pics must also be copied into static/images. Then, the picture name/number can be added to the house_images list in the frontend.py file.
