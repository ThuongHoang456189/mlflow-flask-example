#!/bin/bash

# Start MLflow server in the background
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

# Start the Flask app via Gunicorn
gunicorn --bind 0.0.0.0:8001 app:app