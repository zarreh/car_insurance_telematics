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

