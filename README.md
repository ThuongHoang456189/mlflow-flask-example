# MLflow Classification Project

[![Docker Hub Image](https://hub.docker.com/repository/docker/hoangthuongdev/mlflow-classifier-flask-app-included)](https://hub.docker.com/repository/docker/hoangthuongdev/mlflow-classifier-flask-app-included)
[![GitHub Repository](https://github.com/ThuongHoang456189/mlflow-flask-example.git)](https://github.com/ThuongHoang456189/mlflow-flask-example.git)

This project demonstrates a machine learning workflow for binary classification using MLflow, containerized with Docker, and deployed on AWS EC2 instances with an RDS PostgreSQL database and S3 storage.

## Live Deployments

* **MLflow UI (EC2 Instance 1):** [http://54.254.217.150:5000](http://54.254.217.150:5000)

* **Flask Web Application (EC2 Instance 2):** [http://http://13.212.216.186:8001](http://http://13.212.216.186:8001)

## Project Overview

This project manages a machine learning workflow for a binary classification task, including data generation, model training, hyperparameter tuning, evaluation, registration, and deployment via a Flask web application.

## Key Components

* **Machine Learning Model:** A Logistic Regression model trained on synthetic data for binary classification. MLflow is used for tracking experiments, logging parameters and metrics, and registering the best model.
* **Flask Web Application:** A user-friendly web interface and a RESTful API built with Flask to interact with the deployed MLflow model for predictions.
* **Docker Containerization:** The entire application, including the MLflow server and Flask app, is containerized using Docker for easy deployment and environment consistency.
* **AWS Deployment:** The project is deployed on AWS infrastructure, utilizing:
    * **EC2 (Elastic Compute Cloud):** Virtual servers to run the MLflow server and the Flask application.
    * **RDS (Relational Database Service) PostgreSQL:** A managed PostgreSQL database to store MLflow experiment and run data.
    * **S3 (Simple Storage Service):** Scalable object storage to store MLflow artifacts (e.g., trained models).

## Repository Contents

The GitHub repository likely contains the following:

* `classifier.py`: Python script for data generation, model training, hyperparameter tuning, evaluation, and MLflow integration.
* `app.py`: Python script for the Flask web application to load the MLflow model and serve predictions.
* `templates/`: Directory containing the HTML template (`index.html`) for the Flask web interface.
* `Dockerfile`: Configuration file for building the Docker image of the Flask website.
* `requirements.txt`: List of Python dependencies.
* `start.sh`: Shell script to start the MLflow server and Flask application within the Docker container.
* `README.md`: This file, providing an overview of the project and instructions.
* Potentially other configuration files or scripts.

## Docker Hub Image

The Docker image for this project is available on Docker Hub:

[https://hub.docker.com/repository/docker/hoangthuongdev/mlflow-classifier-flask-app-included](https://hub.docker.com/repository/docker/hoangthuongdev/mlflow-classifier-flask-app-included)


## GitHub Repository

The source code for this project is hosted on GitHub:

[https://github.com/ThuongHoang456189/mlflow-flask-example.git](https://github.com/ThuongHoang456189/mlflow-flask-example.git)

## Deployment

The MLflow UI and Flask web application are deployed on separate AWS EC2 instances. You can access them using the following links:

* **MLflow UI:** [http://54.254.217.150:5000](http://54.254.217.150:5000)
* **Flask Web Application:** [http://http://13.212.216.186:8001](http://http://13.212.216.186:8001)

## Getting Started

To run this project locally (assuming you have Docker installed):

1.  Clone the GitHub repository:
    ```bash
    git clone [https://github.com/ThuongHoang456189/mlflow-flask-example.git](https://github.com/ThuongHoang456189/mlflow-flask-example.git
    cd mlflow-flask-example
    ```
2.  Build the Docker image:
    ```bash
    docker build -t mlflow-classifier-flask-app-included .
    ```
3.  Run the Docker container (this example exposes ports for local testing, but for full functionality with MLflow tracking and artifact storage, you'd need to configure the `MLFLOW_TRACKING_URI` and artifact root):
    ```bash
    docker run -p 8001:8001 -p 5000:5000 mlflow-classifier-flask-app-included
    ```
4.  Access the Flask app at `http://localhost:8001` and the MLflow UI at `http://localhost:5000`.

For full deployment on AWS, please refer to the detailed deployment steps in the project documentation.