# Makefile

# Specify the shell to use
SHELL=/bin/bash

# Environment variables for AWS S3 and PostgreSQL RDS
export MLFLOW_TRACKING_URI=http://localhost:5000
export S3_BUCKET_NAME=mlflow-XXX-XXX-artifacts
export RDS_ENDPOINT=mlflow-database.XXX.XXX.rds.amazonaws.com
export RDS_PORT=5432
export RDS_USERNAME=XXX
export RDS_PASSWORD=XXX
export RDS_DATABASE=mlflow

VENV_DIR=venv

# Default target
all: setup venv install mlflow-server run

# Setup environment variables
setup:
	@echo "Setting up environment variables..."
	@env | grep -E 'MLFLOW_TRACKING_URI|S3_BUCKET_NAME|RDS_' || true

# Create or verify virtual environment
venv:
	@echo "Checking for virtual environment in $(VENV_DIR)..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }; \
	else \
		echo "Virtual environment already exists in $(VENV_DIR)."; \
	fi

# Install dependencies
install: requirements.txt venv
	@echo "Checking for curl..."
	@command -v curl >/dev/null 2>&1 || { echo "Error: curl is not installed. Install it with 'sudo apt install curl'"; exit 1; }
	@echo "Installing dependencies in $(VENV_DIR)..."
	@. $(VENV_DIR)/bin/activate && pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

# Run the MLflow server in the background
mlflow-server: install
	@echo "Starting MLflow server in the background..."
	@. $(VENV_DIR)/bin/activate && mlflow server \
		--backend-store-uri "postgresql://$(RDS_USERNAME):$(RDS_PASSWORD)@$(RDS_ENDPOINT):$(RDS_PORT)/$(RDS_DATABASE)" \
		--default-artifact-root "s3://$(S3_BUCKET_NAME)/mlruns" \
		--host 0.0.0.0 \
		--port 5000 \
		--workers 2 > mlflow_server.log 2>&1 & \
	 echo "MLflow server started with PID $$!"; \
	 sleep 5; \

# Run the classifier.py script
run: mlflow-server
	@echo "Running classifier.py with environment variables..."
	@. $(VENV_DIR)/bin/activate && python3 classifier.py || { echo "Failed to run classifier.py. Check classifier output or mlflow_server.log."; exit 1; }

# Clean up virtual environment and logs
clean:
	@echo "Removing virtual environment in $(VENV_DIR) and logs..."
	@rm -rf $(VENV_DIR) mlflow_server.log
	@pkill -f "mlflow server" || true
	@echo "Stopped any running MLflow server processes."

# Help target
help:
	@echo "Available targets:"
	@echo "  make all        - Run setup, create venv, install dependencies, start MLflow server, and run classifier.py"
	@echo "  make setup      - Display environment variables"
	@echo "  make venv       - Create or verify virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make mlflow-server - Start MLflow server in the background"
	@echo "  make run        - Run classifier.py"
	@echo "  make clean      - Remove virtual environment, logs, and stop MLflow server"
	@echo "  make help       - Show this help message"

.PHONY: all setup venv install mlflow-server run clean help