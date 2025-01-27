import joblib
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
# Load the trained model
try:
    model = joblib.load("Neural_network_iris.pkl")
except FileNotFoundError:
    model = None
    print("Model file not found. Ensure 'Neural_network_iris.pkl' is present.")

    # Load the Iris dataset for target names
iris = load_iris()
required_params = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


# Extract parameters from request
features = [5.1,3.5,1.4,0.2]
prediction = model.predict(features)
probabilities = prediction[0]
print(probabilities)
print()
print(prediction)