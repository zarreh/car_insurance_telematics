#!/bin/bash

# Docker build and run script for Car Insurance Telematics project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="car-insurance-telematics"
CONTAINER_NAME="car-telematics-app"
BUILD_ONLY=false
RUN_JUPYTER=false

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Car Insurance Telematics Docker Build & Run Script

Usage: $0 [OPTIONS] [COMMAND]

OPTIONS:
    -h, --help          Show this help message
    -b, --build-only    Only build the image, don't run container
    -j, --jupyter       Run Jupyter notebook service instead of main app
    -n, --name NAME     Set custom container name (default: car-telematics-app)

COMMANDS:
    build              Build the Docker image
    run                Run the container interactively
    train              Run model training
    infer-sample       Run inference with sample data
    infer-batch        Run batch inference
    lint               Run code linting
    stop               Stop and remove container
    clean              Remove container and image

EXAMPLES:
    $0 build                    # Build the image
    $0 run                      # Run container interactively
    $0 train                    # Train models
    $0 infer-sample             # Run inference with sample data
    $0 --jupyter                # Start Jupyter notebook server
    $0 clean                    # Clean up containers and images

EOF
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
    print_status "Image built successfully!"
}

# Function to run container
run_container() {
    local command=${1:-"bash"}
    
    # Stop and remove existing container if it exists
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Stopping and removing existing container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME > /dev/null 2>&1 || true
        docker rm $CONTAINER_NAME > /dev/null 2>&1 || true
    fi
    
    print_status "Running container: $CONTAINER_NAME"
    docker run -it --rm \
        --name $CONTAINER_NAME \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/model_registry:/app/model_registry" \
        -v "$(pwd)/logs:/app/logs" \
        $IMAGE_NAME $command
}

# Function to run Jupyter
run_jupyter() {
    if docker ps -a --format 'table {{.Names}}' | grep -q "^car-telematics-jupyter$"; then
        print_warning "Stopping existing Jupyter container"
        docker stop car-telematics-jupyter > /dev/null 2>&1 || true
        docker rm car-telematics-jupyter > /dev/null 2>&1 || true
    fi
    
    print_status "Starting Jupyter notebook server on http://localhost:8888"
    docker run -it --rm \
        --name car-telematics-jupyter \
        -p 8888:8888 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/model_registry:/app/model_registry" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/notebooks:/app/notebooks" \
        -v "$(pwd)/car_insurance_telematics:/app/car_insurance_telematics" \
        $IMAGE_NAME bash -c "poetry run pip install jupyter && poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
}

# Function to stop container
stop_container() {
    if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_status "Stopping container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME
    else
        print_warning "Container $CONTAINER_NAME is not running"
    fi
}

# Function to clean up
clean_up() {
    print_status "Cleaning up containers and images"
    
    # Stop and remove containers
    for container in $CONTAINER_NAME car-telematics-jupyter; do
        if docker ps -a --format 'table {{.Names}}' | grep -q "^${container}$"; then
            docker stop $container > /dev/null 2>&1 || true
            docker rm $container > /dev/null 2>&1 || true
            print_status "Removed container: $container"
        fi
    done
    
    # Remove image
    if docker images --format 'table {{.Repository}}' | grep -q "^${IMAGE_NAME}$"; then
        docker rmi $IMAGE_NAME
        print_status "Removed image: $IMAGE_NAME"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        -j|--jupyter)
            RUN_JUPYTER=true
            shift
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        build)
            build_image
            exit 0
            ;;
        run)
            build_image
            run_container "bash"
            exit 0
            ;;
        train)
            build_image
            run_container "train"
            exit 0
            ;;
        infer-sample)
            build_image
            run_container "infer-sample"
            exit 0
            ;;
        infer-batch)
            build_image
            run_container "infer-batch"
            exit 0
            ;;
        lint)
            build_image
            run_container "lint"
            exit 0
            ;;
        stop)
            stop_container
            exit 0
            ;;
        clean)
            clean_up
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default behavior: build and run
build_image

if [[ "$BUILD_ONLY" == "true" ]]; then
    print_status "Build completed. Use '$0 run' to start the container."
elif [[ "$RUN_JUPYTER" == "true" ]]; then
    run_jupyter
else
    run_container
fi
