import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from itertools import product

# Set the tracking URI explicitly
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_tracking_uri("postgresql://mlflowuser:Mlflow123!@mlflow-classifier-db.creo6q2s4q75.ap-southeast-1.rds.amazonaws.com:5432/mlflow_db")

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}

# Initialize MLflow
experiment_name = "classification_experiment"
mlflow.set_experiment(experiment_name)

# Verify the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment:
    print(f"Experiment '{experiment_name}' (ID: {experiment.experiment_id}) set successfully.")
else:
    print(f"Failed to set experiment '{experiment_name}'.")

best_f1 = 0
best_run_id = None

# Hyperparameter tuning
for C, solver in product(param_grid['C'], param_grid['solver']):
    try:
        with mlflow.start_run():
            # Train model
            model = LogisticRegression(C=C, solver=solver, random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = mlflow.active_run().info.run_id
    except Exception as e:
        print(f"Error during run with C={C}, solver={solver}: {str(e)}")

# Register the best model
try:
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "BestClassificationModel")
        print(f"Registered best model with run_id {best_run_id} as 'BestClassificationModel'.")
    else:
        print("No best run found to register.")
except Exception as e:
    print(f"Error registering model: {str(e)}")

# Compare results
experiment = mlflow.get_experiment_by_name("classification_experiment")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    print("\nModel Comparison:")
    print(runs[['run_id', 'params.C', 'params.solver', 'metrics.accuracy', 'metrics.f1_score']])
else:
    print("Experiment not found, cannot compare runs.")