"""
Model Trainer
Orchestrates the entire training pipeline for telematics ML models
"""

import json
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from claim_probability_model import ClaimProbabilityModel
from claim_severity_model import ClaimSeverityModel
from feature_engineer import FeatureEngineer
from model_evaluator import ModelEvaluator

from model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates the complete model training pipeline"""

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
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_registry_dir, exist_ok=True)

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.model_evaluator = ModelEvaluator(output_dir)
        self.model_registry = ModelRegistry(model_registry_dir)

        # Training configuration
        self.config = {"test_size": 0.2, "val_size": 0.2, "random_state": 42, "claim_threshold": 0.5}

        logger.info(f"ModelTrainer initialized with data from {data_path}")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data and prepare features and targets

        Returns:
            Tuple of (features, claim_probability_target, claim_amount_target)
        """
        logger.info("Loading and preparing data...")

        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} records")

        # Create features
        features = self.feature_engineer.create_features(df)
        logger.info(f"Created {features.shape[1]} features")

        # Create targets
        # For claim probability: binary target
        claim_probability_target = df["has_claim"].astype(int)

        # For claim severity: only consider records with claims
        claim_amount_target = df["claim_amount"]

        return features, claim_probability_target, claim_amount_target

    def split_data(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets

        Args:
            features: Feature dataframe
            target: Target series

        Returns:
            Dictionary containing split datasets
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            target,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=target if target.dtype == "int" else None,
        )

        # Second split: train vs val
        val_size_adjusted = self.config["val_size"] / (1 - self.config["test_size"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.config["random_state"],
            stratify=y_temp if y_temp.dtype == "int" else None,
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    def train_claim_probability_model(self, model_type: str = "random_forest", **model_params) -> Dict[str, Any]:
        """
        Train the claim probability model

        Args:
            model_type: Type of model to train
            **model_params: Additional model parameters

        Returns:
            Dictionary containing model and results
        """
        logger.info("Training claim probability model...")

        # Load and prepare data
        features, claim_target, _ = self.load_and_prepare_data()

        # Split data
        data_splits = self.split_data(features, claim_target)

        # Initialize model
        model = ClaimProbabilityModel(model_type=model_type, **model_params)

        # Train model
        train_metrics = model.train(
            data_splits["X_train"], data_splits["y_train"], data_splits["X_val"], data_splits["y_val"]
        )

        # Evaluate on test set
        test_predictions = model.predict(data_splits["X_test"])
        test_probabilities = model.predict_proba(data_splits["X_test"])[:, 1]

        # Generate comprehensive evaluation
        evaluation_results = self.model_evaluator.evaluate_classification(
            data_splits["y_test"], test_predictions, test_probabilities, model_name=f"claim_probability_{model_type}"
        )

        # Save model to registry
        model_info = {
            "model_type": "claim_probability",
            "algorithm": model_type,
            "metrics": {**train_metrics, **evaluation_results["metrics"]},
            "feature_importance": model.feature_importance.to_dict("records"),
            "training_date": datetime.now().isoformat(),
        }

        model_path = self.model_registry.save_model(model, "claim_probability", model_info)

        # Save training results
        results = {
            "model_path": model_path,
            "train_metrics": train_metrics,
            "test_metrics": evaluation_results["metrics"],
            "feature_importance": model.feature_importance,
            "data_splits_info": {
                "train_size": len(data_splits["X_train"]),
                "val_size": len(data_splits["X_val"]),
                "test_size": len(data_splits["X_test"]),
            },
        }

        # Save results to file
        results_path = os.path.join(self.output_dir, f"claim_probability_{model_type}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Claim probability model training completed. Results saved to {results_path}")

        return {"model": model, "results": results, "evaluation": evaluation_results}

    def train_claim_severity_model(self, model_type: str = "random_forest", **model_params) -> Dict[str, Any]:
        """
        Train the claim severity model

        Args:
            model_type: Type of model to train
            **model_params: Additional model parameters

        Returns:
            Dictionary containing model and results
        """
        logger.info("Training claim severity model...")

        # Load and prepare data
        features, _, claim_amounts = self.load_and_prepare_data()

        # Filter for only claims with positive amounts
        mask = claim_amounts > 0
        features_with_claims = features[mask]
        claim_amounts_positive = claim_amounts[mask]

        logger.info(f"Training on {len(features_with_claims)} records with positive claim amounts")

        # Split data
        data_splits = self.split_data(features_with_claims, claim_amounts_positive)

        # Initialize model
        model = ClaimSeverityModel(model_type=model_type, **model_params)

        # Train model
        train_metrics = model.train(
            data_splits["X_train"], data_splits["y_train"], data_splits["X_val"], data_splits["y_val"]
        )

        # Evaluate on test set
        test_predictions = model.predict(data_splits["X_test"])

        # Generate comprehensive evaluation
        evaluation_results = self.model_evaluator.evaluate_regression(
            data_splits["y_test"], test_predictions, model_name=f"claim_severity_{model_type}"
        )

        # Save model to registry
        model_info = {
            "model_type": "claim_severity",
            "algorithm": model_type,
            "metrics": {**train_metrics, **evaluation_results["metrics"]},
            "feature_importance": model.feature_importance.to_dict("records"),
            "training_date": datetime.now().isoformat(),
        }

        model_path = self.model_registry.save_model(model, "claim_severity", model_info)

        # Save training results
        results = {
            "model_path": model_path,
            "train_metrics": train_metrics,
            "test_metrics": evaluation_results["metrics"],
            "feature_importance": model.feature_importance,
            "data_splits_info": {
                "train_size": len(data_splits["X_train"]),
                "val_size": len(data_splits["X_val"]),
                "test_size": len(data_splits["X_test"]),
            },
        }

        # Save results to file
        results_path = os.path.join(self.output_dir, f"claim_severity_{model_type}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Claim severity model training completed. Results saved to {results_path}")

        return {"model": model, "results": results, "evaluation": evaluation_results}

    def train_all_models(
        self, probability_model_type: str = "random_forest", severity_model_type: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Train both claim probability and severity models

        Args:
            probability_model_type: Model type for probability prediction
            severity_model_type: Model type for severity prediction

        Returns:
            Dictionary containing both models and results
        """
        logger.info("Starting full model training pipeline...")

        # Train claim probability model
        probability_results = self.train_claim_probability_model(probability_model_type)

        # Train claim severity model
        severity_results = self.train_claim_severity_model(severity_model_type)

        # Create combined summary
        summary = {
            "training_date": datetime.now().isoformat(),
            "data_path": self.data_path,
            "models": {
                "claim_probability": {
                    "type": probability_model_type,
                    "test_auc": probability_results["results"]["test_metrics"]["test_roc_auc"],
                    "model_path": probability_results["results"]["model_path"],
                },
                "claim_severity": {
                    "type": severity_model_type,
                    "test_rmse": severity_results["results"]["test_metrics"]["test_rmse"],
                    "test_r2": severity_results["results"]["test_metrics"]["test_r2"],
                    "model_path": severity_results["results"]["model_path"],
                },
            },
        }

        # Save summary
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Full model training completed. Summary saved to {summary_path}")

        return {"probability_model": probability_results, "severity_model": severity_results, "summary": summary}

    def compare_models(
        self, model_types: list = ["random_forest", "gradient_boosting", "logistic_regression"]
    ) -> Dict[str, Any]:
        """
        Train and compare multiple model types

        Args:
            model_types: List of model types to compare

        Returns:
            Comparison results
        """
        logger.info(f"Comparing models: {model_types}")

        comparison_results = {"claim_probability": {}, "claim_severity": {}}

        # Compare claim probability models
        for model_type in model_types:
            if model_type in ["random_forest", "gradient_boosting", "logistic_regression"]:
                logger.info(f"Training claim probability model: {model_type}")
                results = self.train_claim_probability_model(model_type)
                comparison_results["claim_probability"][model_type] = {
                    "test_auc": results["results"]["test_metrics"]["test_roc_auc"],
                    "test_pr_auc": results["results"]["test_metrics"]["test_pr_auc"],
                    "model_path": results["results"]["model_path"],
                }

        # Compare claim severity models (exclude logistic regression)
        severity_model_types = [m for m in model_types if m != "logistic_regression"]
        for model_type in severity_model_types:
            logger.info(f"Training claim severity model: {model_type}")
            results = self.train_claim_severity_model(model_type)
            comparison_results["claim_severity"][model_type] = {
                "test_rmse": results["results"]["test_metrics"]["test_rmse"],
                "test_r2": results["results"]["test_metrics"]["test_r2"],
                "test_mae": results["results"]["test_metrics"]["test_mae"],
                "model_path": results["results"]["model_path"],
            }

        # Save comparison results
        comparison_path = os.path.join(self.output_dir, "model_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2)

        logger.info(f"Model comparison completed. Results saved to {comparison_path}")

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

    # Or compare different model types
    # comparison = trainer.compare_models(['random_forest', 'gradient_boosting', 'logistic_regression'])
