# Telematics Machine Learning Pipeline

A comprehensive machine learning pipeline for telematics data analysis, focusing on insurance claim prediction.

## Overview

This pipeline provides end-to-end functionality for:
- **Claim Probability Prediction**: Binary classification to predict likelihood of insurance claims
- **Claim Severity Prediction**: Regression to predict claim amounts
- **Risk Scoring**: Comprehensive risk assessment for drivers/trips

## Project Structure

```
modeling/
├── feature_engineer.py          # Feature engineering module
├── claim_probability_model.py   # Classification model for claim likelihood
├── claim_severity_model.py      # Regression model for claim amounts
├── model_trainer.py            # Training pipeline orchestrator
├── model_evaluator.py          # Model evaluation and visualization
├── model_registry.py           # Model versioning and management
├── inference_pipeline.py       # Production inference pipeline
├── train_models.py             # Main training script
├── run_inference.py            # Main inference script
└── README.md                   # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the processed data file in the correct location:
```
data/processed/processed_trips_1200_drivers.csv
```

## Quick Start

### Training Models

1. **Train default models (Random Forest for both)**:
```bash
python train_models.py
```

2. **Train specific model types**:
```bash
python train_models.py --prob-model gradient_boosting --severity-model ridge
```

3. **Compare multiple models**:
```bash
python train_models.py --compare-models
```

### Running Inference

1. **Run inference on a file**:
```bash
python run_inference.py --input-file path/to/your/data.csv
```

2. **Test with sample data**:
```bash
python run_inference.py --use-sample-data
```

3. **Single prediction demo**:
```bash
python run_inference.py --single-prediction
```

4. **Export for production**:
```bash
python run_inference.py --export-production
```

## Features

### Feature Engineering
- **52+ engineered features** across 6 categories:
  - Basic trip metrics (duration, distance)
  - Speed-related features
  - Driving behavior indicators
  - Time-based risk factors
  - Environmental conditions
  - Interaction features

### Models Supported

**Classification (Claim Probability)**:
- Random Forest
- Gradient Boosting
- Logistic Regression

**Regression (Claim Severity)**:
- Random Forest
- Gradient Boosting
- Linear Regression
- Ridge Regression
- Lasso Regression

### Model Management
- Automatic versioning
- Model registry with metadata
- Performance tracking
- Easy model promotion to production

### Evaluation Metrics

**Classification**:
- ROC AUC, PR AUC
- Precision, Recall, F1
- Calibration metrics
- Confusion matrix

**Regression**:
- RMSE, MAE, R²
- MAPE
- Residual analysis
- Prediction intervals

## API Usage

### Training Pipeline

```python
from model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    data_path='data/processed/processed_trips_1200_drivers.csv',
    output_dir='data/ml_results',
    model_registry_dir='model_registry'
)

# Train all models
results = trainer.train_all_models(
    probability_model_type='random_forest',
    severity_model_type='gradient_boosting'
)
```

### Inference Pipeline

```python
from inference_pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline()

# Single prediction
trip_data = {
    'trip_duration_minutes': 45.5,
    'trip_distance_km': 32.1,
    'max_speed_kmh': 95.0,
    # ... other features
}
prediction = pipeline.predict_single(trip_data)

# Batch prediction
predictions = pipeline.predict_from_file('input_data.csv')
```

### Model Registry

```python
from model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# List all models
models = registry.list_models()

# Load specific model
model = registry.load_model('claim_probability', version='v2')

# Compare versions
comparison = registry.compare_versions('claim_probability')
```

## Output Structure

### Training Outputs
```
data/ml_results/
├── claim_probability_random_forest_results.json
├── claim_severity_random_forest_results.json
├── training_summary.json
├── model_comparison.json
└── [evaluation plots]
```

### Model Registry
```
model_registry/
├── registry.json
├── claim_probability/
│   ├── v1/
│   │   ├── model.pkl
│   │   └── model_info.json
│   └── v2/
└── claim_severity/
```

### Inference Outputs
```
data/ml_results/
├── predictions_YYYYMMDD_HHMMSS.csv
└── predictions_YYYYMMDD_HHMMSS_summary.json
```

## Command Line Options

### train_models.py
- `--data-path`: Path to processed data CSV
- `--output-dir`: Directory for results
- `--model-registry-dir`: Model registry location
- `--prob-model`: Claim probability model type
- `--severity-model`: Claim severity model type
- `--compare-models`: Compare multiple models

### run_inference.py
- `--input-file`: Input CSV file path
- `--output-file`: Output predictions path
- `--model-registry-dir`: Model registry location
- `--single-prediction`: Demo single prediction
- `--use-sample-data`: Use built-in sample data
- `--no-uncertainty`: Disable uncertainty estimates
- `--export-production`: Export for deployment

## Performance Considerations

- Feature engineering is optimized for batch processing
- Models support parallel training where applicable
- Inference pipeline handles large datasets efficiently
- Model registry enables quick model switching

## Best Practices

1. **Data Quality**: Ensure input data matches expected schema
2. **Model Selection**: Use cross-validation results to choose models
3. **Monitoring**: Track prediction drift in production
4. **Retraining**: Schedule regular model updates with new data
5. **Version Control**: Use model registry for all deployments

## Troubleshooting

### Common Issues

1. **"No models found"**: Train models first using `train_models.py`
2. **Missing features**: Ensure all required columns are in input data
3. **Memory errors**: Process large files in batches
4. **Import errors**: Check Python path and dependencies

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new features:
1. Update `feature_engineer.py` with new feature logic
2. Add feature descriptions and importance hints
3. Update model training to handle new features
4. Test with existing models
5. Document changes in this README

## License

This project is proprietary. All rights reserved.

## Contact

For questions or support, contact the ML team.
