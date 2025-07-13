# Telematics Risk Modeling Module

Machine learning models for predicting insurance claim probability and severity from driving behavior data.

## Overview

This module implements a two-stage modeling approach:
1. **Claim Probability Model**: Binary classification to predict if a trip will result in a claim
2. **Claim Severity Model**: Regression to predict claim amount given that a claim occurs

## Models

### Claim Probability Model (`claim_probability_model.py`)
- **Algorithm**: XGBoost Classifier with probability calibration
- **Features**: 57 engineered features from trip data
- **Handles**: Class imbalance (scale_pos_weight), feature scaling, cross-validation
- **Output**: Calibrated probability of claim (0-1)

### Claim Severity Model (`claim_severity_model.py`)
- **Algorithm**: XGBoost Regressor with log transformation
- **Features**: Same 57 features, trained only on trips with claims
- **Handles**: Skewed claim amounts, feature scaling, uncertainty estimation
- **Output**: Expected claim amount in dollars

## Key Components

### Feature Engineering (`feature_engineer.py`)
Creates 57 features across categories:
- Basic trip metrics (duration, distance, speed)
- Driving behavior (hard braking, acceleration, phone use)
- Temporal patterns (time of day, rush hour, weekend)
- Environmental factors (weather, traffic density)
- Statistical aggregations (speed variance, percentiles)

### Model Training (`model_trainer.py`)
```bash
# Train all models
python -m car_insurance_telematics.modeling.model_trainer

# Train specific model type
python -m car_insurance_telematics.modeling.model_trainer --model-type xgboost
```

### Model Evaluation (`model_evaluator.py`)
- Classification metrics: AUC-ROC, Precision-Recall, F1
- Regression metrics: RMSE, MAE, R², MAPE
- Visualization: ROC curves, feature importance, residual plots

### Model Registry (`model_registry.py`)
- Version control for trained models
- Metadata tracking (metrics, parameters, timestamps)
- Easy model loading for inference

### Inference Pipeline (`inference_pipeline.py`)
```python
from car_insurance_telematics.modeling import InferencePipeline

# Initialize
pipeline = InferencePipeline()

# Single prediction
result = pipeline.predict_single(trip_data)
# Returns: claim_probability, expected_severity, risk_category, risk_score

# Batch prediction
results = pipeline.predict_from_file("trips.csv", "predictions.json")
```

## Usage Examples

### Training
```python
from car_insurance_telematics.modeling import ModelTrainer

trainer = ModelTrainer(data_path="data/processed/trips.csv")
results = trainer.train_all_models()
```

### Feature Engineering
```python
from car_insurance_telematics.modeling import FeatureEngineer

fe = FeatureEngineer()
features = fe.create_features(trip_df)
importance = fe.get_feature_importance_interpretation(feature_importance_df)
```

### Risk Assessment
```python
# Risk categories based on claim probability:
# - Low: < 2%
# - Medium: 2-5%
# - High: 5-10%
# - Very High: > 10%

# Risk score: 0-100 scale for easy interpretation
risk_score = min(100, claim_probability * 1000)
```

## Model Performance

Typical performance metrics (on synthetic data):
- **Claim Probability**: AUC-ROC ~0.85-0.90
- **Claim Severity**: R² ~0.65-0.75, RMSE ~$2,500

## Dependencies

- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- joblib >= 1.3.0

## Files

- `claim_probability_model.py`: Binary classification model
- `claim_severity_model.py`: Regression model for claim amounts
- `feature_engineer.py`: Feature creation and engineering
- `model_trainer.py`: Training pipeline and hyperparameter tuning
- `model_evaluator.py`: Model evaluation and visualization
- `model_registry.py`: Model versioning and storage
- `inference_pipeline.py`: Production inference pipeline
