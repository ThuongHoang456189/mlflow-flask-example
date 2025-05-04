import mlflow
import mlflow.sklearn
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from itertools import product

# Set the tracking URI from environment variable
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset information
print("\nDataset Information:")
print(f"Total samples in X: {X.shape[0]}")
print(f"Total samples in y: {len(y)}")
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution in y:")
print(f" - Number of y=0: {np.sum(y == 0)}")
print(f" - Number of y=1: {np.sum(y == 1)}")

# Convert X to a DataFrame for better visualization
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
print("\nExample of dataset (first 5 rows):")
print(X_df.head(5).to_string(index=False))
print("\nExample of target (first 5 values):")
print(y[:5])

# Define expanded hyperparameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear'],
    'tol': [1e-4, 1e-3],
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
for params in product(
    param_grid['C'],
    param_grid['solver'],
    param_grid['tol']
):
    C, solver, tol = params
    try:
        with mlflow.start_run():
            # Train model
            model = LogisticRegression(
                C=C,
                solver=solver,
                tol=tol,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_param("tol", tol)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log model to S3
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = mlflow.active_run().info.run_id
    except Exception as e:
        print(f"Error during run with params {params}: {str(e)}")

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
    print(runs[['run_id', 'params.C', 'params.solver', 'params.tol', 'metrics.accuracy', 'metrics.f1_score']])
else:
    print("Experiment not found, cannot compare runs.")