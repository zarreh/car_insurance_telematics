# Car Insurance Telematics Risk Assessment System

A machine learning system for assessing driver risk and predicting insurance claims based on telematics data. This system processes trip data from vehicle sensors to predict claim probability and potential claim severity.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Docker Installation (Recommended)](#docker-installation-recommended)
  - [Local Development Setup](#local-development-setup)
- [Quick Start](#quick-start)
  - [Using Docker](#using-docker)
  - [Local Environment](#local-environment)
- [Docker Usage](#docker-usage)
  - [Docker Commands](#docker-commands)
  - [Docker Compose](#docker-compose)
  - [Volume Mounts](#volume-mounts)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Experimetation](#experimentation)
  - [Notebooks Overview](#notebooks-overview)
  - [Experimental Design](#experimental-design)
  - [Model Performance](#model-performance)
  - [Key Insights](#key-insights)

## Overview

This system analyzes driving behavior data to:
- Predict the probability of insurance claims
- Estimate potential claim amounts
- Categorize drivers by risk level
- Provide interpretable risk factors

The pipeline processes raw telematics data (GPS, accelerometer, speed) into features that capture driving behavior patterns, then uses ensemble machine learning models to make predictions.

## Features

- **Comprehensive Feature Engineering**: 50+ engineered features capturing driving behavior, time patterns, and risk indicators
- **Dual Model Architecture**: Separate models for claim probability and severity
- **Model Registry**: Version control and management for trained models
- **Batch & Real-time Inference**: Support for both batch processing and single-trip predictions
- **Risk Categorization**: Automatic classification into risk tiers (Low/Medium/High/Very High)
- **Interpretability**: Feature importance analysis and risk factor explanations
- **Modular Design**: Easy to extend with new features or models
- **🐳 Docker Support**: Fully containerized environment with Docker and Docker Compose
- **📦 Easy Deployment**: One-command setup for training, inference, and development
- **🎯 Multiple Interfaces**: Command-line tools, Python API, and Jupyter notebooks

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Trip      │────▶│ Feature          │────▶│ ML Models       │
│   JSON Data     │     │ Engineering      │     │ - Probability   │
└─────────────────┘     └──────────────────┘     │ - Severity      │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ Risk Assessment │
                                                  │ & Predictions   │
                                                  └─────────────────┘
```

## Installation

### Docker Installation (Recommended)

🐳 **The easiest way to get started is using Docker**. This ensures consistent environments and eliminates dependency issues.

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (version 20.0+)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, for advanced usage)

#### Quick Setup
```bash
# Clone the repository
git clone https://github.com/zarreh/car-insurance-telematics.git
cd car-insurance-telematics

# Make the docker script executable
chmod +x docker.sh

# Build and run interactively (one command!)
./docker.sh run
```

#### Alternative Docker Commands
```bash
# Build the image
docker build -t car-insurance-telematics .

# Run with volume mounts
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_registry:/app/model_registry \
  -v $(pwd)/logs:/app/logs \
  car-insurance-telematics bash
```

### Local Development Setup

#### Prerequisites
- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)

#### Setup Steps
```bash
# Clone the repository
git clone https://github.com/zarreh/car-insurance-telematics.git
cd car-insurance-telematics

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell

# Create necessary directories
mkdir -p data/{raw,processed,ml_results}
mkdir -p model_registry logs
```

## Quick Start

### Using Docker

🚀 **Get started in seconds with Docker**:

```bash
# Train models
./docker.sh train

# Run inference with sample data
./docker.sh infer-sample

# Run batch inference
./docker.sh infer-batch

# Start Jupyter notebooks for development
./docker.sh --jupyter
# Access at http://localhost:8888

# Interactive shell
./docker.sh run
```

### Local Environment

#### 1. Process Raw Data
```bash
# Process JSON trip files
python -m car_insurance_telematics.preprocessing.preprocess

# Or process specific files
python -m car_insurance_telematics.preprocessing.preprocess --input-dir data/raw --output-dir data/processed
```

#### 2. Train Models
```bash
# Train all models with default settings
python -m car_insurance_telematics.modeling.train_models

# Or use make commands
make train
```

#### 3. Run Inference
```bash
# Run inference on processed data
python -m car_insurance_telematics.modeling.run_inference --input-file data/processed/processed_trips_1200_drivers.csv

# Or use sample data for testing
python -m car_insurance_telematics.modeling.run_inference --use-sample-data

# Or use make commands
make infer
make infer-batch
```

## Docker Usage

### Docker Commands

The `docker.sh` script provides convenient commands for all operations:

```bash
# Show help
./docker.sh --help

# Build and run commands
./docker.sh build              # Build the Docker image only
./docker.sh run                # Run container interactively
./docker.sh train              # Train ML models
./docker.sh infer-sample       # Run inference with sample data
./docker.sh infer-batch        # Run batch inference
./docker.sh lint               # Run code linting and formatting

# Development commands
./docker.sh --jupyter          # Start Jupyter notebook server
./docker.sh --name my-app      # Use custom container name

# Cleanup commands
./docker.sh stop               # Stop running container
./docker.sh clean              # Remove containers and images
```

### Docker Compose

For more advanced usage with services:

```bash
# Start main application
docker-compose up --build

# Start with Jupyter notebooks
docker-compose --profile jupyter up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Volume Mounts

The Docker setup automatically mounts the following directories:

| Host Directory | Container Directory | Purpose |
|----------------|-------------------|---------|
| `./data/` | `/app/data/` | Training data and results |
| `./model_registry/` | `/app/model_registry/` | Saved models and metadata |
| `./logs/` | `/app/logs/` | Application logs |
| `./notebooks/` | `/app/notebooks/` | Jupyter notebooks (dev mode) |

### Docker Files Overview

| File | Purpose |
|------|---------|
| `Dockerfile` | Main container definition with Python 3.12, Poetry, and ML dependencies |
| `.dockerignore` | Excludes unnecessary files from build context |
| `docker-compose.yml` | Multi-service orchestration with volumes and networking |
| `docker.sh` | Convenience script with common Docker operations |

### Manual Docker Commands

If you prefer manual Docker commands:

```bash
# Build image
docker build -t car-insurance-telematics .

# Run training
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_registry:/app/model_registry \
  -v $(pwd)/logs:/app/logs \
  car-insurance-telematics train

# Run inference
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_registry:/app/model_registry \
  car-insurance-telematics infer-sample

# Interactive development
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_registry:/app/model_registry \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/car_insurance_telematics:/app/car_insurance_telematics \
  car-insurance-telematics bash

# Jupyter notebook server
docker run --rm -it \
  -p 8888:8888 \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/data:/app/data \
  car-insurance-telematics bash -c "poetry run pip install jupyter && poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

## Usage

### Data Processing

The system expects raw trip data in JSON format with the following structure:

```json
{
  "driver_id": "D001",
  "trip_id": "T001",
  "start_time": "2024-01-15T08:30:00",
  "end_time": "2024-01-15T09:15:00",
  "gps_data": [...],
  "acceleration_events": [...],
  "speed_data": [...]
}
```

Process raw data:
```python
from car_insurance_telematics.preprocessing import DataProcessor

processor = DataProcessor(config_path="config.yaml")
processor.process_directory("data/raw", "data/processed")
```

### Model Training

Train models with custom parameters:
```python
from car_insurance_telematics.modeling import ModelTrainer

trainer = ModelTrainer(data_path="data/processed/processed_trips.csv")

# Train specific model type
results = trainer.train_claim_probability_model(model_type="gradient_boosting")

# Or train all models
all_results = trainer.train_all_models()

# Compare different models
comparison = trainer.compare_models(["random_forest", "gradient_boosting", "logistic_regression"])
```

### Inference

#### Single Trip Prediction
```python
from car_insurance_telematics.modeling import InferencePipeline
import pandas as pd

# Initialize pipeline
pipeline = InferencePipeline()

# Single trip data
trip_data = {
    'driver_id': 'D001',
    'trip_duration_minutes': 45,
    'trip_distance_km': 40,
    'avg_speed_kmh': 53,
    'max_speed_kmh': 85,
    'hard_braking_count': 3,
    'hard_acceleration_count': 2,
    'sharp_turn_count': 1,
    # ... other features
}

# Get prediction
result = pipeline.predict_single(trip_data)
print(f"Claim Probability: {result['claim_probability']:.2%}")
print(f"Risk Category: {result['risk_category']}")
```

#### Batch Prediction
```python
# Predict for multiple trips
results = pipeline.predict_from_file(
    "data/processed/new_trips.csv",
    output_file="predictions.json"
)
```

## Project Structure

```
car_insurance_telematics/
├── car_insurance_telematics/           # Main Python package
│   ├── __init__.py
│   ├── data_generation/               # Sample data generation
│   │   ├── __init__.py
│   │   └── json_sample_generation.py
│   ├── preprocess/                    # Data preprocessing
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration management
│   │   ├── preprocess.py              # Main preprocessing pipeline
│   │   ├── data_loader.py             # Data loading utilities
│   │   ├── feature_engineering.py     # Feature engineering
│   │   ├── main.py                    # CLI entry point
│   │   └── utils.py                   # Utility functions
│   └── modeling/                      # ML modeling package
│       ├── __init__.py
│       ├── feature_engineer.py        # Feature engineering for ML
│       ├── claim_probability_model.py # Classification model
│       ├── claim_severity_model.py    # Regression model
│       ├── model_trainer.py           # Training pipeline
│       ├── model_evaluator.py         # Model evaluation
│       ├── model_registry.py          # Model versioning
│       ├── inference_pipeline.py      # Inference pipeline
│       ├── train_models.py            # Training script
│       └── run_inference.py           # Inference script
├── data/                              # Data directory
│   ├── raw/                           # Raw JSON trip files
│   ├── processed/                     # Processed CSV files
│   ├── archive/                       # Archived datasets
│   └── ml_results/                    # Model outputs and evaluations
├── model_registry/                    # Saved models and metadata
│   ├── registry.json                  # Model registry metadata
│   ├── claim_probability/             # Probability models
│   └── claim_severity/                # Severity models
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 00_telematics_eda.ipynb       # Exploratory data analysis
│   ├── 01_feature_engineering.ipynb  # Feature engineering
│   ├── 02_xgboost_claim_prediction.ipynb # Claim prediction
│   ├── 03_xgboost_claim_severity.ipynb   # Claim severity
│   └── files/                         # Notebook output files
├── logs/                              # Application logs
├── tests/                             # Unit tests
├── 🐳 Docker Files:
├── Dockerfile                         # Main container definition
├── .dockerignore                      # Docker build exclusions
├── docker-compose.yml                 # Multi-service orchestration
├── docker.sh                          # Docker convenience script
├── 📁 Configuration Files:
├── pyproject.toml                     # Poetry configuration & dependencies
├── poetry.lock                        # Locked dependency versions
├── Makefile                           # Build commands
└── README.md                          # This documentation
```

## Models

### Claim Probability Model
- **Type**: Binary Classification
- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
- **Features**: 57 engineered features from trip data
- **Output**: Probability of claim (0-1)
- **Metrics**: AUC-ROC, Precision, Recall, F1-Score

### Claim Severity Model
- **Type**: Regression
- **Algorithms**: Random Forest, Gradient Boosting
- **Features**: Same 57 features, trained only on trips with claims
- **Output**: Expected claim amount ($)
- **Metrics**: RMSE, MAE, R², MAPE

### Risk Categorization
Based on claim probability:
- **Low Risk**: < 2% claim probability
- **Medium Risk**: 2-5% claim probability
- **High Risk**: 5-10% claim probability
- **Very High Risk**: > 10% claim probability

## API Reference

### Feature Engineering

```python
from car_insurance_telematics.modeling import FeatureEngineer

fe = FeatureEngineer()
features = fe.create_features(trip_dataframe)
feature_names = fe.get_feature_names()
```

### Model Training

```python
from car_insurance_telematics.modeling import ModelTrainer

trainer = ModelTrainer(data_path="path/to/data.csv")
results = trainer.train_all_models()
```

### Inference Pipeline

```python
from car_insurance_telematics.modeling import InferencePipeline

pipeline = InferencePipeline()

# Single prediction
result = pipeline.predict_single(trip_dict)

# Batch prediction
results = pipeline.predict_batch(trips_dataframe)

# From file
output_file = pipeline.predict_from_file("input.csv", "output.json")
```

## Configuration

Configuration is managed through `config.yaml`:

```yaml
data_processing:
  chunk_size: 1000
  output_format: "parquet"

feature_engineering:
  speed_bins: [0, 30, 60, 90, 120, 200]
  distance_bins: [0, 5, 20, 50, 100, 1000]

modeling:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

  claim_probability:
    algorithms: ["random_forest", "gradient_boosting"]
    hyperparameters:
      random_forest:
        n_estimators: 100
        max_depth: 10

  claim_severity:
    algorithms: ["random_forest", "gradient_boosting"]
    hyperparameters:
      random_forest:
        n_estimators: 100
        max_depth: 15
```

## Development

### Setting up Development Environment

#### Docker Development (Recommended)

```bash
# Interactive development with auto-reload
./docker.sh run

# Jupyter notebook development
./docker.sh --jupyter
# Access at http://localhost:8888

# Mount source code for live editing
docker run --rm -it \
  -v $(pwd)/car_insurance_telematics:/app/car_insurance_telematics \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/notebooks:/app/notebooks \
  car-insurance-telematics bash

# Code formatting and linting in container
./docker.sh lint
```

#### Local Development

```bash
# Install development dependencies
poetry install --with dev

# Activate virtual environment
poetry shell

# Install pre-commit hooks (optional)
pre-commit install

# Run tests
pytest

# Run linting and formatting
make lint
# Or manually:
poetry run autoflake car_insurance_telematics --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
poetry run black car_insurance_telematics --line-length 120 -q
poetry run isort car_insurance_telematics
```

### Development Workflow

#### Using Docker for Consistent Development

```bash
# 1. Make code changes locally
# 2. Test in container
./docker.sh train  # Test training pipeline

# 3. Run inference tests
./docker.sh infer-sample

# 4. Notebook development
./docker.sh --jupyter

# 5. Format code
./docker.sh lint
```

#### Available Make Commands

```bash
make lint          # Format and lint code
make train         # Train models locally
make infer         # Run sample inference
make infer-batch   # Run batch inference
```

### Adding New Features

1. **Feature Engineering**: Update `FeatureEngineer.create_features()` in `feature_engineer.py`
2. **Feature Documentation**: Add feature interpretation in `get_feature_importance_interpretation()`
3. **Testing**: Update tests in `tests/test_feature_engineer.py`
4. **Validation**: Test with both Docker and local environments

### Adding New Models

1. **Model Class**: Create new model class inheriting from base model
2. **Implementation**: Implement `train()`, `predict()`, and `get_feature_importance()` methods
3. **Registration**: Register in `ModelTrainer` class
4. **Docker Testing**: Test new model with `./docker.sh train`
5. **Documentation**: Update model documentation in README

### Docker Development Tips

- **Live Code Changes**: Mount source code volume for immediate updates
- **Data Persistence**: Use volume mounts to persist data between container runs
- **Multiple Environments**: Use different container names for different experiments
- **Debugging**: Use `./docker.sh run` for interactive debugging sessions
- **Resource Management**: Monitor Docker resource usage for large datasets

## Testing

### Docker Testing

```bash
# Run tests in container
docker run --rm \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/car_insurance_telematics:/app/car_insurance_telematics \
  car-insurance-telematics bash -c "python -m pytest tests/"

# Test training pipeline
./docker.sh train

# Test inference pipeline
./docker.sh infer-sample
```

### Local Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=car_insurance_telematics

# Run specific test file
pytest tests/test_feature_engineer.py

# Run integration tests
pytest tests/integration/
```

## Deployment

### Production Deployment with Docker

#### 1. Build Production Image

```bash
# Build optimized production image
docker build -t car-insurance-telematics:prod .

# Or use specific version
docker build -t car-insurance-telematics:v1.0.0 .
```

#### 2. Run in Production

```bash
# Production training (with volume mounts for data persistence)
docker run -d \
  --name telematics-training \
  -v /path/to/production/data:/app/data \
  -v /path/to/production/models:/app/model_registry \
  -v /path/to/production/logs:/app/logs \
  car-insurance-telematics:prod train

# Production inference service
docker run -d \
  --name telematics-inference \
  -p 8000:8000 \
  -v /path/to/production/data:/app/data \
  -v /path/to/production/models:/app/model_registry \
  car-insurance-telematics:prod infer-batch
```

#### 3. Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  telematics-app:
    image: car-insurance-telematics:prod
    volumes:
      - /production/data:/app/data:ro
      - /production/models:/app/model_registry
      - /production/logs:/app/logs
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
```

#### 4. Container Orchestration

For Kubernetes deployment:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: car-insurance-telematics
spec:
  replicas: 1
  selector:
    matchLabels:
      app: telematics
  template:
    metadata:
      labels:
        app: telematics
    spec:
      containers:
      - name: telematics
        image: car-insurance-telematics:prod
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: model-volume
          mountPath: /app/model_registry
```

### Environment Variables

The container supports these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python module search path |
| `PYTHONUNBUFFERED` | `1` | Ensure stdout/stderr are unbuffered |
| `MODEL_REGISTRY_PATH` | `/app/model_registry` | Path to model storage |
| `DATA_PATH` | `/app/data` | Path to data directory |
| `LOG_LEVEL` | `INFO` | Logging level |

### Health Checks

Add health checks to your Docker deployment:

```dockerfile
# Add to Dockerfile for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import car_insurance_telematics; print('OK')" || exit 1
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8
- Add type hints for all functions
- Write docstrings for all classes and methods
- Add unit tests for new functionality
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with scikit-learn, pandas, and numpy
- Inspired by real-world telematics insurance applications
- Thanks to all contributors

## Contact

For questions or support, please contact:
- Email: ali@zarreh.ai
- Issues: zarreh.ai

## Notebooks Overview

### 1. Feature Engineering (`01_feature_engineering.ipynb`)

**Purpose**: Transform raw telematics data into meaningful features for machine learning models.

**Key Features Created**:
- **Driver Behavior Metrics**: Harsh braking/acceleration events, speed patterns, phone usage
- **Risk Indicators**: Composite risk scores, driving intensity measures
- **Temporal Features**: Night driving, rush hour patterns, trip timing analysis
- **Aggregated Statistics**: Driver-level summaries from trip-level data

**Outputs**:
- `driver_level_features.csv`: 1,200 drivers × 55 features
- `trip_level_features.csv`: 17,819 trips × 36 features
- `feature_descriptions.csv`: Feature documentation

**Key Techniques**:
- Statistical aggregation (mean, std, sum, max)
- Risk scoring algorithms
- Feature selection and engineering
- Data quality assessment

### 2. Claim Prediction (`02_xgboost_claim_prediction.ipynb`)

**Purpose**: Build a binary classification model to predict claim probability.

**Model Architecture**:
- **Algorithm**: XGBoost Classifier
- **Optimization**: Optuna hyperparameter tuning (20 trials)
- **Validation**: 3-fold stratified cross-validation
- **Metrics**: F1-score, ROC-AUC, Precision, Recall

**Key Results**:
- **Best F1-Score**: 0.0562 (optimized model)
- **ROC-AUC**: 0.6891 (test set)
- **Feature Importance**: Composite risk score, harsh driving intensity
- **Model Interpretability**: SHAP values and feature analysis

**Hyperparameter Optimization**:
- Search space: 9 XGBoost parameters
- Objective: Maximize F1-score
- Pruning: Early stopping for poor trials
- Best parameters automatically selected

### 3. Claim Severity (`03_xgboost_claim_severity.ipynb`)

**Purpose**: Build a regression model to predict claim amounts for drivers with claims.

**Model Architecture**:
- **Algorithm**: XGBoost Regressor
- **Optimization**: Optuna hyperparameter tuning (20 trials)
- **Validation**: 3-fold cross-validation
- **Metrics**: RMSE, MAE, R², MAPE

**Key Results**:
- **Best RMSE**: $1,847 (optimized model)
- **R² Score**: 0.485 (explains 48.5% of variance)
- **MAE**: $1,456 (mean absolute error)
- **Target Range**: $600 - $16,000 claim amounts

**Business Applications**:
- Premium calculation and pricing
- Risk-based customer segmentation
- Claims reserving and budgeting

## Experimental Design

### Hyperparameter Optimization with Optuna

Both models use **Optuna** for systematic hyperparameter optimization:

**Search Parameters**:
- `n_estimators`: 100-1000 trees
- `max_depth`: 3-10 levels
- `learning_rate`: 0.01-0.3
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`: 1e-8 to 10 (L1 regularization)
- `reg_lambda`: 1e-8 to 10 (L2 regularization)
- `min_child_weight`: 1-10
- `gamma`: 1e-8 to 1.0

**Optimization Strategy**:
- **Classification**: Maximize F1-score (handles class imbalance)
- **Regression**: Minimize RMSE (standard for claim amounts)
- **Trials**: 20 iterations per model
- **Pruning**: Automatic early stopping for poor performers

### Feature Selection and Engineering

**Driver-Level Aggregations**:
- **53 features** derived from trip-level data
- **Risk Scoring**: Composite algorithms combining multiple risk factors
- **Behavioral Patterns**: Speed, harsh events, phone usage, timing
- **Experience Indicators**: Mileage, trip frequency, consistency

**Feature Categories**:
1. **Basic Statistics**: Trip counts, distances, durations
2. **Speed Behavior**: Average, maximum, variance, risk flags
3. **Aggressive Driving**: Harsh events per mile/minute
4. **Distraction**: Phone usage patterns and excessive use flags
5. **Temporal Risk**: Night driving, rush hour exposure
6. **Data Quality**: GPS accuracy, signal quality scores

## Model Performance

### Claim Prediction (Classification)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| F1-Score | 0.0366 | 0.0562 | +53.6% |
| ROC-AUC | 0.5240 | 0.6891 | +31.5% |
| Precision | 0.1037 | 0.0833 | -19.7% |
| Recall | 0.0222 | 0.0444 | +100.0% |

### Claim Severity (Regression)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| RMSE | $2,156 | $1,847 | -14.3% |
| MAE | $1,678 | $1,456 | -13.2% |
| R² | 0.398 | 0.485 | +21.9% |
| MAPE | 52.8% | 46.2% | -12.5% |

## Key Insights

### Risk Factors for Claims

**Top Predictive Features**:
1. **Composite Risk Score**: Overall driving risk assessment
2. **Harsh Driving Intensity**: Combined aggressive driving events
3. **Speed Risk Score**: Speeding and excessive speed patterns
4. **Total Distance**: Higher mileage increases exposure
5. **Night Driving**: Increased risk during nighttime hours

### Business Applications

**Insurance Pricing**:
- Risk-based premium calculation
- Dynamic pricing based on driving behavior
- Customer segmentation for targeted products

**Claims Management**:
- Early identification of high-risk drivers
- Proactive intervention and coaching programs
- Accurate claims reserving and budgeting

**Product Development**:
- Usage-based insurance (UBI) programs
- Telematics-enabled discounts
- Behavioral modification incentives

## Technical Requirements

### Dependencies

```python
# Core ML Libraries
xgboost>=1.6.0
scikit-learn>=1.0.0
optuna>=3.0.0

# Data Processing
pandas>=1.4.0
numpy>=1.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
joblib>=1.1.0
```

### Hardware Recommendations

- **CPU**: 4+ cores for parallel processing
- **RAM**: 8GB+ for large datasets
- **Storage**: 2GB+ for data and models
- **Runtime**: ~30 minutes per optimization (20 trials)

## Usage Instructions

### 1. Data Preparation
```bash
# Ensure data files are in correct locations
data/processed/processed_trips_1200_drivers.csv
```

### 2. Feature Engineering
```bash
# Run notebook 01 to create features
jupyter notebook 01_feature_engineering.ipynb
```

### 3. Model Training
```bash
# Train claim prediction model
jupyter notebook 02_xgboost_claim_prediction.ipynb

# Train claim severity model  
jupyter notebook 03_xgboost_claim_severity.ipynb
```

### 4. Model Deployment
```python
# Load trained models
import joblib
claim_model = joblib.load('models/optimized_claim_prediction_model.pkl')
severity_model = joblib.load('models/optimized_claim_severity_model.pkl')

# Make predictions
claim_prob = claim_model.predict_proba(features)[:, 1]
claim_amount = severity_model.predict(features)
```

## Model Interpretability

### Feature Importance Analysis

Both models provide detailed feature importance rankings:
- **SHAP values** for individual prediction explanations
- **Permutation importance** for robust feature ranking
- **Partial dependence plots** for feature relationship analysis

### Business Rules Integration

Models can be combined with business rules:
```python
# Risk-based pricing example
def calculate_premium(base_premium, claim_prob, claim_severity):
    risk_multiplier = 1 + (claim_prob * claim_severity / 1000)
    return base_premium * risk_multiplier
```

## 🐳 Docker Quick Reference

### Essential Commands

| Purpose | Command | Description |
|---------|---------|-------------|
| **Setup** | `./docker.sh build` | Build the Docker image |
| **Training** | `./docker.sh train` | Train ML models |
| **Inference** | `./docker.sh infer-sample` | Test with sample data |
| **Batch** | `./docker.sh infer-batch` | Process large datasets |
| **Development** | `./docker.sh --jupyter` | Start Jupyter notebooks |
| **Interactive** | `./docker.sh run` | Open interactive shell |
| **Cleanup** | `./docker.sh clean` | Remove containers/images |

### File Summary

| File | Purpose | Key Features |
|------|---------|--------------|
| `Dockerfile` | Container definition | Python 3.12, Poetry, ML libs |
| `docker-compose.yml` | Service orchestration | Volume mounts, networking |
| `docker.sh` | Convenience script | One-command operations |
| `.dockerignore` | Build optimization | Excludes unnecessary files |

### Quick Start Reminder

```bash
# Clone and start in 3 commands
git clone https://github.com/zarreh/car-insurance-telematics.git
cd car-insurance-telematics
./docker.sh train  # Builds image and trains models!
```

### Production Checklist

- ✅ Docker image builds successfully
- ✅ Training pipeline completes without errors
- ✅ Inference works with sample data
- ✅ Volumes mounted for data persistence
- ✅ Environment variables configured
- ✅ Health checks implemented
- ✅ Container resource limits set
- ✅ Logging and monitoring configured

## Future Enhancements

### Model Improvements
- **Ensemble Methods**: Combine multiple algorithms
- **Deep Learning**: Neural networks for complex patterns
- **Time Series**: Temporal modeling of driving behavior
- **Causal Inference**: Understanding cause-effect relationships

### Feature Engineering
- **Geospatial Features**: Location-based risk factors
- **Weather Integration**: Environmental driving conditions
- **Vehicle Telematics**: Engine, brake, and sensor data
- **External Data**: Traffic, road conditions, demographics

### Production Deployment
- **Real-time Scoring**: API endpoints for live predictions
- **Model Monitoring**: Performance tracking and drift detection
- **A/B Testing**: Controlled model comparison
- **Automated Retraining**: Continuous model updates

