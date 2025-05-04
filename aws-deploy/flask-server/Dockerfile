FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/

EXPOSE 8001

# Set environment variable for MLflow tracking URI
ENV MLFLOW_TRACKING_URI=http://54.254.217.150:5000

# Run the Flask app with Gunicorn, 2 workers, using gevent (async)
CMD ["gunicorn", "--workers", "2", "--worker-class", "gevent", "--bind", "0.0.0.0:8001", "app:app"]