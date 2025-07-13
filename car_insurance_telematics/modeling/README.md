# Car Insurance Telematics Risk Assessment System

A machine learning system for assessing driver risk and predicting insurance claims based on telematics data. This system processes trip data from vehicle sensors to predict claim probability and potential claim severity.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
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

### Prerequisites
- Python 3.8+
- Poetry (for dependency management)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/car-insurance-telematics.git
cd car-insurance-telematics
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Create necessary directories:
```bash
mkdir -p data/{raw,processed,ml_results}
mkdir -p model_registry
```

## Quick Start

### 1. Process Raw Data
```bash
# Process JSON trip files
python -m car_insurance_telematics.preprocessing.preprocess

# Or process specific files
python -m car_insurance_telematics.preprocessing.preprocess --input-dir data/raw --output-dir data/processed
```

### 2. Train Models
```bash
# Train all models with default settings
python -m car_insurance_telematics.modeling.model_trainer

# Or train specific model types
python -m car_insurance_telematics.modeling.model_trainer --model-types random_forest gradient_boosting
```

### 3. Run Inference
```bash
# Run inference on processed data
python run_inference.py --input-file data/processed/processed_trips.csv

# Or use sample data for testing
python run_inference.py --use-sample-data
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
├── car_insurance_telematics/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── preprocess.py      # Main preprocessing pipeline
│   │   └── data_loader.py     # Data loading utilities
│   └── modeling/
│       ├── __init__.py
│       ├── feature_engineer.py        # Feature engineering
│       ├── claim_probability_model.py # Classification model
│       ├── claim_severity_model.py    # Regression model
│       ├── model_trainer.py           # Training pipeline
│       ├── model_evaluator.py         # Model evaluation
│       ├── model_registry.py          # Model versioning
│       └── inference_pipeline.py      # Inference pipeline
├── data/
│   ├── raw/                   # Raw JSON trip files
│   ├── processed/             # Processed CSV files
│   └── ml_results/            # Model outputs and evaluations
├── model_registry/            # Saved models and metadata
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── pyproject.toml            # Poetry configuration
├── README.md                 # This file
└── requirements.txt          # Alternative dependencies list
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

```bash
# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 car_insurance_telematics/
black car_insurance_telematics/
```

### Adding New Features

1. Update `FeatureEngineer.create_features()` in `feature_engineer.py`
2. Add feature interpretation in `get_feature_importance_interpretation()`
3. Update tests in `tests/test_feature_engineer.py`

### Adding New Models

1. Create new model class inheriting from base model
2. Implement `train()`, `predict()`, and `get_feature_importance()` methods
3. Register in `ModelTrainer` class
4. Add tests

## Testing

Run the test suite:

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
- Email: telematics-ml@yourcompany.com
- Issues: https://github.com/your-org/car-insurance-telematics/issues
