"""
Model Trainer
Orchestrates the training of telematics ML models
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from car_insurance_telematics.modeling.claim_probability_model import ClaimProbabilityModel
from car_insurance_telematics.modeling.claim_severity_model import ClaimSeverityModel
from car_insurance_telematics.modeling.feature_engineer import FeatureEngineer
from car_insurance_telematics.modeling.model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split

from car_insurance_telematics.modeling.model_registry import ModelRegistry

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/model_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates the training of telematics ML models"""

    def __init__(self, data_path: str, output_dir: str = "data/ml_results", model_registry_dir: str = "model_registry"):
        """
        Initialize the model trainer

        Args:
            data_path: Path to the processed data CSV file
            output_dir: Directory to save training results
            model_registry_dir: Directory for model registry
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_registry_dir = model_registry_dir

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_registry_dir, exist_ok=True)

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator(output_dir=self.output_dir)
        self.registry = ModelRegistry(registry_dir=self.model_registry_dir)

        logger.info(f"ModelTrainer initialized with data from {data_path}")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data and prepare features and targets

        Returns:
            Tuple of (features, claim_probability_target, claim_severity_target)
        """
        logger.info("Loading and preparing data...")

        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} records")

        # Create features
        features = self.feature_engineer.create_features(df)
        logger.info(f"Created {len(features.columns)} features")

        # Handle target columns
        # Check if target columns exist, if not create synthetic ones
        if "has_claim" not in df.columns:
            logger.warning("'has_claim' column not found. Creating synthetic target based on risk factors...")
            # Create synthetic claim probability based on risk indicators
            risk_score = (
                (df.get("hard_braking_count", pd.Series([0] * len(df))) > 2).astype(int) * 0.2
                + (df.get("hard_acceleration_count", pd.Series([0] * len(df))) > 2).astype(int) * 0.2
                + (df.get("max_speed_kmh", pd.Series([80] * len(df))) > 120).astype(int) * 0.3
                + (df.get("night_driving_minutes", pd.Series([0] * len(df))) > 20).astype(int) * 0.15
                + (df.get("phone_use_minutes", pd.Series([0] * len(df))) > 5).astype(int) * 0.15
            )
            # Add some randomness
            random_factor = np.random.uniform(-0.1, 0.1, size=len(df))
            claim_probability = (risk_score + random_factor).clip(0, 1)
            # Convert to binary outcome
            df["has_claim"] = (claim_probability > np.random.uniform(0, 1, size=len(df))).astype(int)
            logger.info(
                f"Created synthetic has_claim with {df['has_claim'].sum()} positive cases ({df['has_claim'].mean():.2%})"
            )

        if "claim_amount" not in df.columns:
            logger.warning("'claim_amount' column not found. Creating synthetic target...")
            # Create synthetic claim amounts for those with claims
            base_amount = np.random.lognormal(8, 1.5, size=len(df))  # Log-normal distribution
            # Scale by risk factors
            severity_multiplier = 1 + (
                (df["max_speed_kmh"] if "max_speed_kmh" in df.columns else 80) / 200
                + (df["total_aggressive_events"] if "total_aggressive_events" in df.columns else 0) / 10
                + (df["adverse_weather"] if "adverse_weather" in df.columns else 0) * 0.5
            )
            df["claim_amount"] = np.where(df["has_claim"] == 1, (base_amount * severity_multiplier).clip(100, 50000), 0)
            logger.info(
                f"Created synthetic claim_amount with mean ${df[df['has_claim']==1]['claim_amount'].mean():.2f}"
            )

        # Extract targets
        claim_probability_target = df["has_claim"].astype(int)
        claim_severity_target = df["claim_amount"].astype(float)

        # For severity model, we only use records with claims
        severity_mask = claim_probability_target == 1
        logger.info(f"Found {severity_mask.sum()} records with claims for severity modeling")

        return features, claim_probability_target, claim_severity_target

    def train_claim_probability_model(self, model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train the claim probability model

        Args:
            model_type: Type of model to train

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training claim probability model ({model_type})...")

        # Load and prepare data
        features, claim_target, _ = self.load_and_prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, claim_target, test_size=0.2, random_state=42, stratify=claim_target
        )

        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Class distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

        # Initialize and train model
        model = ClaimProbabilityModel(model_type=model_type)
        model.train(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)

        # Evaluate model
        train_metrics = self.evaluator.evaluate_classification(y_train, train_pred, train_proba, prefix="train")
        test_metrics = self.evaluator.evaluate_classification(y_test, test_pred, test_proba, prefix="test")

        # Get feature importance
        feature_importance = model.feature_importance

        # Create evaluation plots
        self.evaluator.plot_classification_results(
            y_test, test_pred, test_proba, model_name=f"claim_probability_{model_type}"
        )

        # Save model to registry
        model_path = self.registry.save_model(
            model=model,
            name="claim_probability",
            model_type=model_type,
            metrics=test_metrics,
            feature_names=self.feature_engineer.get_feature_names(),
        )

        # Prepare results
        # Prepare results
        results = {
            "model_type": model_type,
            "model_path": model_path,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance.to_dict("records") if feature_importance is not None else [],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "positive_rate": float(claim_target.mean()),
        }

        # Save results
        results_path = os.path.join(self.output_dir, f"claim_probability_{model_type}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Claim probability model training completed. Test AUC: {test_metrics['test_roc_auc']:.4f}")

        return results

    def train_claim_severity_model(self, model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train the claim severity model

        Args:
            model_type: Type of model to train

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training claim severity model ({model_type})...")

        # Load and prepare data
        features, claim_target, severity_target = self.load_and_prepare_data()

        # Filter for claims only
        claim_mask = claim_target == 1
        features_with_claims = features[claim_mask]
        severity_with_claims = severity_target[claim_mask]

        if len(features_with_claims) < 50:
            logger.warning(f"Only {len(features_with_claims)} claims available. Results may be unreliable.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_with_claims, severity_with_claims, test_size=0.2, random_state=42
        )

        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Claim amount range - Train: ${y_train.min():.2f} - ${y_train.max():.2f}")

        # Initialize and train model
        model = ClaimSeverityModel(model_type=model_type)
        model.train(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Evaluate model
        train_metrics = self.evaluator.evaluate_regression(y_train, train_pred, prefix="train")
        test_metrics = self.evaluator.evaluate_regression(y_test, test_pred, prefix="test")

        # Get feature importance
        feature_importance = model.feature_importance

        # Create evaluation plots
        self.evaluator.plot_regression_results(y_test, test_pred, model_name=f"claim_severity_{model_type}")

        # Save model to registry
        model_path = self.registry.save_model(
            model=model,
            name="claim_severity",
            model_type=model_type,
            metrics=test_metrics,
            feature_names=self.feature_engineer.get_feature_names(),
        )

        # Prepare results
        results = {
            "model_type": model_type,
            "model_path": model_path,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance.to_dict("records") if feature_importance is not None else [],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "mean_claim_amount": float(severity_with_claims.mean()),
            "median_claim_amount": float(severity_with_claims.median()),
        }
        # Save results
        results_path = os.path.join(self.output_dir, f"claim_severity_{model_type}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Claim severity model training completed. Test RMSE: ${test_metrics['test_rmse']:.2f}")

        return results

    def train_all_models(
        self, probability_model_type: str = "random_forest", severity_model_type: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Train both claim probability and severity models

        Args:
            probability_model_type: Model type for claim probability
            severity_model_type: Model type for claim severity

        Returns:
            Dictionary with all training results
        """
        logger.info("Starting full model training pipeline...")

        start_time = datetime.now()

        # Train claim probability model
        logger.info("Training claim probability model...")
        probability_results = self.train_claim_probability_model(probability_model_type)

        # Train claim severity model
        logger.info("Training claim severity model...")
        severity_results = self.train_claim_severity_model(severity_model_type)

        # Create summary
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        summary = {
            "training_completed": end_time.isoformat(),
            "training_duration_seconds": training_duration,
            "models": {
                "claim_probability": {
                    "model_type": probability_model_type,
                    "test_auc": probability_results["test_metrics"]["test_roc_auc"],
                    "model_path": probability_results["model_path"],
                },
                "claim_severity": {
                    "model_type": severity_model_type,
                    "test_rmse": severity_results["test_metrics"]["test_rmse"],
                    "test_r2": severity_results["test_metrics"]["test_r2"],
                    "model_path": severity_results["model_path"],
                },
            },
        }

        # Save summary
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"All models trained successfully in {training_duration:.1f} seconds")

        return {
            "probability_model": {"type": probability_model_type, "results": probability_results},
            "severity_model": {"type": severity_model_type, "results": severity_results},
            "summary": summary,
        }

    def compare_models(
        self, model_types: list = ["random_forest", "gradient_boosting", "logistic_regression"]
    ) -> Dict[str, Any]:
        """
        Compare different model types

        Args:
            model_types: List of model types to compare

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing models: {model_types}")

        comparison_results = {"claim_probability": {}, "claim_severity": {}}

        # Compare claim probability models
        for model_type in model_types:
            try:
                results = self.train_claim_probability_model(model_type)
                comparison_results["claim_probability"][model_type] = {
                    "test_auc": results["test_metrics"]["test_roc_auc"],
                    "test_pr_auc": results["test_metrics"]["test_pr_auc"],
                    "test_f1": results["test_metrics"]["test_f1"],
                }
            except Exception as e:
                logger.error(f"Error training {model_type} for claim probability: {str(e)}")

        # Compare claim severity models (only regression-capable models)
        regression_models = [m for m in model_types if m != "logistic_regression"]
        for model_type in regression_models:
            try:
                results = self.train_claim_severity_model(model_type)
                comparison_results["claim_severity"][model_type] = {
                    "test_rmse": results["test_metrics"]["test_rmse"],
                    "test_r2": results["test_metrics"]["test_r2"],
                    "test_mae": results["test_metrics"]["test_mae"],
                }
            except Exception as e:
                logger.error(f"Error training {model_type} for claim severity: {str(e)}")

        # Save comparison results
        comparison_path = os.path.join(self.output_dir, "model_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2)

        # Create comparison plots
        self.evaluator.plot_model_comparison(comparison_results)

        logger.info("Model comparison completed")

        return comparison_results


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(
        data_path="data/processed/processed_trips_1200_drivers.csv",
        output_dir="data/ml_results",
        model_registry_dir="model_registry",
    )

    # Train all models
    results = trainer.train_all_models()

    # Compare different model types
    comparison = trainer.compare_models()
