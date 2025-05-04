# Specify the shell to use
SHELL=/bin/bash

# Define variables
IMAGE_NAME := hoangthuongdev/mlflow-classifier-flask-app:latest
CONTAINER_PORT := 8001
HOST_PORT := 8001

# Default target
all: install-docker pull-image run-container

# Install Docker on Ubuntu
install-docker:
	@echo "Installing Docker on Ubuntu..."
	@sudo apt-get update
	@sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
	@curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
	@echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	@sudo apt-get update
	@sudo apt-get install -y docker-ce docker-ce-cli containerd.io
	@sudo systemctl start docker
	@sudo systemctl enable docker
	@sudo usermod -aG docker $$(whoami)
	@echo "Docker installed successfully."

# Pull the image from Docker Hub
pull-image:
	@echo "Pulling image $(IMAGE_NAME) from Docker Hub..."
	@sudo docker pull $(IMAGE_NAME)
	@echo "Image $(IMAGE_NAME) pulled successfully."

# Run the Docker container (normal operation)
run-container:
	@echo "Running container from image $(IMAGE_NAME)..."
	@sudo docker run -d -p $(HOST_PORT):$(CONTAINER_PORT) --network host $(IMAGE_NAME)
	@echo "Container started on port $(HOST_PORT)."

# Debug the Docker container with /bin/bash (host network)
debug-container: pull-image
	@echo "Checking if Docker is running..."
	@sudo systemctl is-active --quiet docker || { echo "Error: Docker is not running. Start it with 'sudo systemctl start docker'"; exit 1; }
	@echo "Starting container in debug mode with /bin/bash (host network)..."
	@sudo docker run -it --network host --entrypoint /bin/bash $(IMAGE_NAME)
	@echo "Debug session started. Exit the shell to stop the container."

# Debug the Docker container with /bin/bash (bridge network)
debug-container-bridge: pull-image
	@echo "Checking if Docker is running..."
	@sudo systemctl is-active --quiet docker || { echo "Error: Docker is not running. Start it with 'sudo systemctl start docker'"; exit 1; }
	@echo "Starting container in debug mode with /bin/bash (bridge network)..."
	@sudo docker run -it -p $(HOST_PORT):$(CONTAINER_PORT) --entrypoint /bin/bash $(IMAGE_NAME)
	@echo "Debug session started. Exit the shell to stop the container."

# Stop and remove the container (optional cleanup)
stop-container:
	@echo "Stopping and removing any existing container..."
	@sudo docker ps -q --filter "ancestor=$(IMAGE_NAME)" | xargs -r sudo docker stop || true
	@sudo docker ps -a -q --filter "ancestor=$(IMAGE_NAME)" | xargs -r sudo docker rm || true
	@echo "Existing containers stopped and removed."

# Clean up Docker (stop, remove container, and remove image)
clean: stop-container
	@echo "Cleaning up Docker image $(IMAGE_NAME)..."
	@sudo docker images -q $(IMAGE_NAME) | xargs -r sudo docker rmi || true
	@echo "Docker cleanup completed."

# Help target
help:
	@echo "Available targets:"
	@echo "  make all               - Install Docker, pull image, and run container"
	@echo "  make install-docker    - Install Docker on Ubuntu"
	@echo "  make pull-image        - Pull the $(IMAGE_NAME) image from Docker Hub"
	@echo "  make run-container     - Run the container with port mapping"
	@echo "  make debug-container   - Run the container with /bin/bash for debugging (host network)"
	@echo "  make debug-container-bridge - Run the container with /bin/bash for debugging (bridge network)"
	@echo "  make stop-container    - Stop and remove the container"
	@echo "  make clean             - Stop container and remove image"
	@echo "  make help              - Show this help message"

.PHONY: all install-docker pull-image run-container debug-container debug-container-bridge stop-container clean help