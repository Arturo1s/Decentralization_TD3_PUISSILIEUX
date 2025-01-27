from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("Logistic_regression_iris.pkl")
except FileNotFoundError:
    model = None
    print("Model file not found. Ensure 'Logistic_regression_iris.pkl' is present.")

# Load the Iris dataset for target names
iris = load_iris()

@app.route('/predict', methods=['GET'])
def predict():
    """Endpoint for making predictions on the Iris dataset."""
    if not model:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please check the server logs."
        }), 500

    # Required parameters for prediction
    required_params = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    features = []
    for param in required_params:
        value = request.args.get(param)
        if value is None:
            raise ValueError(f"Missing required parameter: {param}")
        features.append(float(value))

    try:
        # Extract parameters from request

        # Convert list to 2D numpy array
        features = np.array([features])

        # Make prediction
        prediction = model.predict_proba(features)
        print("Model output shape:", prediction.shape)

        # Return successful response
        return jsonify({
            "status": "success",
            "probability_scores": prediction[0].tolist()
        })

    except ValueError as ve:
        # Handle missing or invalid inputs
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400

    except Exception as e:
        # General error handling
        return jsonify({
            "status": "error",
            "message": "An error occurred during prediction.",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)