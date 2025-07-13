"""
Claim Probability Model
Binary classification model to predict the likelihood of insurance claims
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimProbabilityModel:
    """Binary classification model for claim probability prediction"""

    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the claim probability model

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_params = kwargs
        self.is_fitted = False
        self.training_metadata = {}

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the underlying ML model"""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)

        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "random_state": 42,
            }
            default_params.update(self.model_params)
            self.model = GradientBoostingClassifier(**default_params)

        elif self.model_type == "logistic_regression":
            default_params = {
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "balanced",
                "random_state": 42,
                "max_iter": 1000,
            }
            default_params.update(self.model_params)
            self.model = LogisticRegression(**default_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the claim probability model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            calibrate: Whether to calibrate probabilities

        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Store training metadata
        self.training_metadata = {
            "model_type": self.model_type,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "class_distribution": y_train.value_counts().to_dict(),
            "training_date": datetime.now().isoformat(),
        }

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Perform cross-validation
        cv_scores = self._cross_validate(X_train_scaled, y_train)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Calibrate probabilities if requested
        if calibrate and self.model_type != "logistic_regression":
            logger.info("Calibrating model probabilities...")
            self.model = CalibratedClassifierCV(self.model, cv=3, method="sigmoid")
            self.model.fit(X_train_scaled, y_train)

        # Extract feature importance
        self._extract_feature_importance(X_train.columns)

        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_metrics = self._evaluate(X_val_scaled, y_val, prefix="val_")

        # Evaluate on training set
        train_metrics = self._evaluate(X_train_scaled, y_train, prefix="train_")

        self.is_fitted = True

        # Combine all metrics
        metrics = {**train_metrics, **val_metrics, **cv_scores, "feature_importance": self.feature_importance}

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict claim probability"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict claim probability scores"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info("Performing cross-validation...")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

        return {"cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std(), "cv_auc_scores": cv_scores.tolist()}

    def _evaluate(self, X: np.ndarray, y: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y, y_proba)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)

        # Calculate accuracy at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        threshold_metrics = {}
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            accuracy = (y_pred_thresh == y).mean()
            threshold_metrics[f"{prefix}accuracy_at_{threshold}"] = accuracy

        return {f"{prefix}roc_auc": roc_auc, f"{prefix}pr_auc": pr_auc, **threshold_metrics}

    def _extract_feature_importance(self, feature_names: pd.Index):
        """Extract feature importance from the model"""
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            # For calibrated models
            if hasattr(self.model, "base_estimator"):
                if hasattr(self.model.base_estimator, "feature_importances_"):
                    importance = self.model.base_estimator.feature_importances_
                elif hasattr(self.model.base_estimator, "coef_"):
                    importance = np.abs(self.model.base_estimator.coef_[0])
                else:
                    importance = np.zeros(len(feature_names))
            else:
                importance = np.zeros(len(feature_names))

        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

    def save(self, filepath: str):
        """Save the model to disk"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance,
            "training_metadata": self.training_metadata,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data["model_type"]
        self.feature_importance = model_data["feature_importance"]
        self.training_metadata = model_data["training_metadata"]
        self.is_fitted = model_data["is_fitted"]
        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "training_metadata": self.training_metadata,
            "model_params": self.model.get_params() if self.model else {},
        }
