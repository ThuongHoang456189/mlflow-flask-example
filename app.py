import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template, jsonify
import numpy as np
import os
import traceback

app = Flask(__name__)

# Set the tracking URI explicitly
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_tracking_uri("http://54.169.242.113:5000")

# Load the best registered model
try:
    print("Attempting to load model: models:/BestClassificationModel/1")
    model = mlflow.sklearn.load_model("models:/BestClassificationModel/1")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()
    model = None

# Validate input features
def validate_features(features):
    try:
        # Check if input is a list or array-like
        if not isinstance(features, (list, np.ndarray)):
            return False, "Input must be a list of numbers"
        
        # Check length
        if len(features) != 20:
            return False, "Exactly 20 features are required"
        
        # Convert to float and check for valid numbers
        features = [float(x) for x in features]
        if any(np.isnan(x) or np.isinf(x) for x in features):
            return False, "Features must be valid numbers (no NaN or Inf)"
        
        return True, features
    except (ValueError, TypeError):
        return False, "All features must be numerical values"

@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        return render_template('error.html', error="Model not loaded. Please check MLflow setup.")
    
    if request.method == 'POST':
        # Get form data
        feature_inputs = [request.form.get(f'feature_{i}') for i in range(20)]
        
        # Validate inputs
        is_valid, result = validate_features(feature_inputs)
        if not is_valid:
            return render_template('index.html', error=result)
        
        # Make prediction
        try:
            features = np.array(result).reshape(1, -1)
            prediction = model.predict(features)[0]
            return render_template('index.html', prediction=prediction, features=feature_inputs)
        except Exception as e:
            return render_template('index.html', error=f"Prediction error: {str(e)}")
    
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'status': 'error', 'error': 'Model not loaded'})
    
    try:
        data = request.get_json()
        features = data.get('features', [])
        
        # Validate input
        is_valid, result = validate_features(features)
        if not is_valid:
            return jsonify({'status': 'error', 'error': result})
        
        # Make prediction
        features = np.array(result).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({
            'status': 'success',
            'prediction': int(prediction)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f"Prediction error: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)