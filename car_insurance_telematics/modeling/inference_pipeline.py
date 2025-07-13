"""
Inference Pipeline
Production-ready inference pipeline for telematics ML models
"""

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from car_insurance_telematics.modeling.feature_engineer import FeatureEngineer
from car_insurance_telematics.modeling.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Production-ready inference pipeline for telematics models"""

    def __init__(self, model_registry_dir: str = "model_registry", output_dir: str = "data/ml_results"):
        """
        Initialize the inference pipeline

        Args:
            model_registry_dir: Directory containing model registry
            output_dir: Directory to save inference results
        """
        self.model_registry_dir = model_registry_dir
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.model_registry = ModelRegistry(model_registry_dir)

        # Load models
        self.claim_prob_model = None
        self.claim_severity_model = None
        self._load_models()

        logger.info("Inference pipeline initialized")

    def _load_models(self, prob_version: Optional[str] = None, severity_version: Optional[str] = None):
        """
        Load models from registry

        Args:
            prob_version: Version of probability model (default: latest)
            severity_version: Version of severity model (default: latest)
        """
        try:
            # Load claim probability model
            self.claim_prob_model = self.model_registry.load_model("claim_probability", prob_version)
            logger.info(f"Loaded claim probability model (version: {prob_version or 'latest'})")

            # Load claim severity model
            self.claim_severity_model = self.model_registry.load_model("claim_severity", severity_version)
            logger.info(f"Loaded claim severity model (version: {severity_version or 'latest'})")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def predict_single(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single trip

        Args:
            trip_data: Dictionary containing trip information

        Returns:
            Dictionary containing predictions
        """
        # Convert to DataFrame for processing
        df = pd.DataFrame([trip_data])

        # Make predictions
        results = self.predict_batch(df)

        # Return first result
        return results[0]

    def predict_batch(self, data: pd.DataFrame, include_uncertainty: bool = True) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of trips

        Args:
            data: DataFrame containing trip data
            include_uncertainty: Whether to include uncertainty estimates

        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Making predictions for {len(data)} trips")

        # Create features
        features = self.feature_engineer.create_features(data)

        # Make claim probability predictions
        claim_probabilities = self.claim_prob_model.predict_proba(features)[:, 1]

        # Make claim severity predictions
        if include_uncertainty and hasattr(self.claim_severity_model, "predict_with_uncertainty"):
            severity_predictions, severity_uncertainty = self.claim_severity_model.predict_with_uncertainty(features)
        else:
            severity_predictions = self.claim_severity_model.predict(features)
            severity_uncertainty = np.zeros_like(severity_predictions)

        # Calculate expected claim amount (probability * severity)
        expected_claim_amounts = claim_probabilities * severity_predictions

        # Create results
        results = []
        for i in range(len(data)):
            result = {
                "driver_id": data.iloc[i].get("driver_id", f"driver_{i}"),
                "trip_id": data.iloc[i].get("trip_id", f"trip_{i}"),
                "claim_probability": float(claim_probabilities[i]),
                "claim_severity_prediction": float(severity_predictions[i]),
                "expected_claim_amount": float(expected_claim_amounts[i]),
                "risk_score": self._calculate_risk_score(claim_probabilities[i], severity_predictions[i]),
                "risk_category": self._categorize_risk(claim_probabilities[i]),
                "prediction_timestamp": datetime.now().isoformat(),
            }

            if include_uncertainty:
                result["severity_uncertainty"] = float(severity_uncertainty[i])
                result["confidence_interval_lower"] = float(
                    max(0, severity_predictions[i] - 2 * severity_uncertainty[i])
                )
                result["confidence_interval_upper"] = float(severity_predictions[i] + 2 * severity_uncertainty[i])

            results.append(result)

        return results

    def predict_from_file(
        self, input_file: str, output_file: Optional[str] = None, include_uncertainty: bool = True
    ) -> str:
        """
        Make predictions from a CSV file

        Args:
            input_file: Path to input CSV file
            output_file: Path to save predictions (optional)
            include_uncertainty: Whether to include uncertainty estimates

        Returns:
            Path to output file
        """
        logger.info(f"Loading data from {input_file}")

        # Load data
        data = pd.read_csv(input_file)

        # Make predictions
        predictions = self.predict_batch(data, include_uncertainty)

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Merge with original data
        result_df = pd.concat(
            [data[["driver_id", "trip_id"]], predictions_df.drop(["driver_id", "trip_id"], axis=1)], axis=1
        )

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"predictions_{timestamp}.csv")

        result_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

        # Generate summary statistics
        self._generate_prediction_summary(result_df, output_file)

        return output_file

    def _calculate_risk_score(self, claim_probability: float, severity_prediction: float) -> float:
        """
        Calculate overall risk score

        Args:
            claim_probability: Probability of claim
            severity_prediction: Predicted claim amount

        Returns:
            Risk score (0-100)
        """
        # Normalize severity to 0-1 range (assuming max claim of $50,000)
        normalized_severity = min(severity_prediction / 50000, 1.0)

        # Weighted combination
        risk_score = (0.6 * claim_probability + 0.4 * normalized_severity) * 100

        return float(min(risk_score, 100))

    def _categorize_risk(self, claim_probability: float) -> str:
        """
        Categorize risk level based on claim probability

        Args:
            claim_probability: Probability of claim

        Returns:
            Risk category
        """
        if claim_probability < 0.1:
            return "Very Low"
        elif claim_probability < 0.25:
            return "Low"
        elif claim_probability < 0.5:
            return "Medium"
        elif claim_probability < 0.75:
            return "High"
        else:
            return "Very High"

    def _generate_prediction_summary(self, predictions_df: pd.DataFrame, output_file: str):
        """Generate summary statistics for predictions"""
        summary_file = output_file.replace(".csv", "_summary.json")

        summary = {
            "prediction_date": datetime.now().isoformat(),
            "total_predictions": len(predictions_df),
            "statistics": {
                "claim_probability": {
                    "mean": float(predictions_df["claim_probability"].mean()),
                    "std": float(predictions_df["claim_probability"].std()),
                    "min": float(predictions_df["claim_probability"].min()),
                    "max": float(predictions_df["claim_probability"].max()),
                    "percentiles": {
                        "25%": float(predictions_df["claim_probability"].quantile(0.25)),
                        "50%": float(predictions_df["claim_probability"].quantile(0.50)),
                        "75%": float(predictions_df["claim_probability"].quantile(0.75)),
                        "90%": float(predictions_df["claim_probability"].quantile(0.90)),
                        "95%": float(predictions_df["claim_probability"].quantile(0.95)),
                    },
                },
                "expected_claim_amount": {
                    "mean": float(predictions_df["expected_claim_amount"].mean()),
                    "std": float(predictions_df["expected_claim_amount"].std()),
                    "min": float(predictions_df["expected_claim_amount"].min()),
                    "max": float(predictions_df["expected_claim_amount"].max()),
                    "total": float(predictions_df["expected_claim_amount"].sum()),
                },
                "risk_score": {
                    "mean": float(predictions_df["risk_score"].mean()),
                    "std": float(predictions_df["risk_score"].std()),
                    "min": float(predictions_df["risk_score"].min()),
                    "max": float(predictions_df["risk_score"].max()),
                },
            },
            "risk_distribution": predictions_df["risk_category"].value_counts().to_dict(),
            "high_risk_count": int((predictions_df["claim_probability"] > 0.5).sum()),
            "model_versions": {
                "claim_probability": self.model_registry.registry["models"]["claim_probability"].get(
                    "latest", "unknown"
                ),
                "claim_severity": self.model_registry.registry["models"]["claim_severity"].get("latest", "unknown"),
            },
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Prediction summary saved to {summary_file}")

    def monitor_predictions(
        self, predictions_df: pd.DataFrame, actual_claims: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Monitor prediction performance and drift

        Args:
            predictions_df: DataFrame with predictions
            actual_claims: DataFrame with actual claim data (optional)

        Returns:
            Monitoring metrics
        """
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "prediction_stats": {
                "total_predictions": len(predictions_df),
                "avg_claim_probability": float(predictions_df["claim_probability"].mean()),
                "high_risk_ratio": float((predictions_df["claim_probability"] > 0.5).mean()),
            },
        }

        # Check for prediction drift
        if hasattr(self, "baseline_stats"):
            drift_metrics = {}

            # Calculate drift for claim probability
            current_mean = predictions_df["claim_probability"].mean()
            baseline_mean = self.baseline_stats["claim_probability_mean"]
            drift_metrics["claim_probability_drift"] = abs(current_mean - baseline_mean) / baseline_mean

            # Calculate drift for severity
            current_severity_mean = predictions_df["claim_severity_prediction"].mean()
            baseline_severity_mean = self.baseline_stats["severity_mean"]
            drift_metrics["severity_drift"] = (
                abs(current_severity_mean - baseline_severity_mean) / baseline_severity_mean
            )

            monitoring_results["drift_metrics"] = drift_metrics

        # If actual claims are provided, calculate performance metrics
        if actual_claims is not None:
            performance_metrics = self._calculate_performance_metrics(predictions_df, actual_claims)
            monitoring_results["performance_metrics"] = performance_metrics

        return monitoring_results

    def _calculate_performance_metrics(
        self, predictions_df: pd.DataFrame, actual_claims: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics comparing predictions to actuals"""
        # Merge predictions with actuals
        merged = predictions_df.merge(
            actual_claims[["trip_id", "has_claim", "claim_amount"]], on="trip_id", how="inner"
        )

        if len(merged) == 0:
            return {}

        # Binary classification metrics
        from sklearn.metrics import (precision_score, recall_score,
                                     roc_auc_score)

        y_true = merged["has_claim"].astype(int)
        y_pred_proba = merged["claim_probability"]
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            "actual_claim_rate": float(y_true.mean()),
            "predicted_claim_rate": float(y_pred.mean()),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }

        # Regression metrics for claims with positive amounts
        claims_mask = merged["has_claim"] == 1
        if claims_mask.sum() > 0:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            actual_amounts = merged.loc[claims_mask, "claim_amount"]
            predicted_amounts = merged.loc[claims_mask, "claim_severity_prediction"]

            metrics["severity_rmse"] = float(np.sqrt(mean_squared_error(actual_amounts, predicted_amounts)))
            metrics["severity_mae"] = float(mean_absolute_error(actual_amounts, predicted_amounts))

        return metrics

    def export_for_production(self, output_dir: str = "production_models"):
        """
        Export models and pipeline for production deployment

        Args:
            output_dir: Directory to export production artifacts
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export configuration
        config = {
            "pipeline_version": "1.0.0",
            "export_date": datetime.now().isoformat(),
            "models": {
                "claim_probability": {
                    "version": self.model_registry.registry["models"]["claim_probability"].get("latest"),
                    "type": "classification",
                },
                "claim_severity": {
                    "version": self.model_registry.registry["models"]["claim_severity"].get("latest"),
                    "type": "regression",
                },
            },
            "feature_engineering": {"version": "1.0.0", "features": self.feature_engineer.get_feature_names()},
        }

        config_path = os.path.join(output_dir, "pipeline_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Copy models to production directory
        for model_type in ["claim_probability", "claim_severity"]:
            self.model_registry.promote_to_production(model_type, config["models"][model_type]["version"])

        # Create deployment instructions
        deployment_instructions = """
# Telematics ML Pipeline Deployment Instructions

## Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, joblib

## Deployment Steps

1. Copy the production_models directory to your production environment
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Load and use the inference pipeline:
   ```python
   from inference_pipeline import InferencePipeline

   # Initialize pipeline
   pipeline = InferencePipeline(
       model_registry_dir='production_models/model_registry',
       output_dir='predictions'
   )

   # Make predictions
   predictions = pipeline.predict_from_file('input_data.csv')
   ```

## API Integration Example

```python
from flask import Flask, request, jsonify
from inference_pipeline import InferencePipeline

app = Flask(__name__)
pipeline = InferencePipeline()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = pipeline.predict_single(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Monitoring
- Monitor prediction drift using the monitor_predictions method
- Set up alerts for significant changes in prediction distributions
- Regularly retrain models with new data

## Model Updates
- Use the ModelRegistry to manage model versions
- Test new models thoroughly before promoting to production
- Maintain backwards compatibility for API consumers
"""

        instructions_path = os.path.join(output_dir, "DEPLOYMENT.md")
        with open(instructions_path, "w") as f:
            f.write(deployment_instructions)

        logger.info(f"Production artifacts exported to {output_dir}")


if __name__ == "__main__":
    # Example usage
    pipeline = InferencePipeline()

    # Example single prediction
    sample_trip = {
        "driver_id": "driver_001",
        "trip_id": "trip_001",
        "trip_duration_minutes": 45.5,
        "trip_distance_km": 32.1,
        "max_speed_kmh": 95.0,
        "avg_speed_kmh": 65.2,
        "hard_braking_count": 2,
        "hard_acceleration_count": 1,
        "night_driving_minutes": 10.0,
        "weekend_driving": 0,
    }

    result = pipeline.predict_single(sample_trip)
    print("Single prediction:", result)
