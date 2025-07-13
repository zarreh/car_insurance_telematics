"""
Claim Severity Model
Regression model to predict claim amounts for insurance claims
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimSeverityModel:
    """Regression model for claim severity (amount) prediction"""

    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the claim severity model

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso')
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
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            self.model = RandomForestRegressor(**default_params)

        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "random_state": 42,
            }
            default_params.update(self.model_params)
            self.model = GradientBoostingRegressor(**default_params)

        elif self.model_type == "linear":
            self.model = LinearRegression()

        elif self.model_type == "ridge":
            default_params = {"alpha": 1.0, "random_state": 42}
            default_params.update(self.model_params)
            self.model = Ridge(**default_params)

        elif self.model_type == "lasso":
            default_params = {"alpha": 1.0, "random_state": 42}
            default_params.update(self.model_params)
            self.model = Lasso(**default_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        log_transform: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the claim severity model

        Args:
            X_train: Training features
            y_train: Training target (claim amounts)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            log_transform: Whether to apply log transformation to target

        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Handle log transformation
        self.log_transform = log_transform
        if log_transform:
            # Add small constant to avoid log(0)
            y_train_transformed = np.log1p(y_train)
            if y_val is not None:
                y_val_transformed = np.log1p(y_val)
        else:
            y_train_transformed = y_train
            if y_val is not None:
                y_val_transformed = y_val

        # Store training metadata
        self.training_metadata = {
            "model_type": self.model_type,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "target_stats": {
                "mean": float(y_train.mean()),
                "std": float(y_train.std()),
                "min": float(y_train.min()),
                "max": float(y_train.max()),
                "median": float(y_train.median()),
            },
            "log_transform": log_transform,
            "training_date": datetime.now().isoformat(),
        }

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Perform cross-validation
        cv_scores = self._cross_validate(X_train_scaled, y_train_transformed)

        # Train the model
        self.model.fit(X_train_scaled, y_train_transformed)

        # Extract feature importance
        self._extract_feature_importance(X_train.columns)

        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_metrics = self._evaluate(X_val_scaled, y_val_transformed, y_val, prefix="val_")

        # Evaluate on training set
        train_metrics = self._evaluate(X_train_scaled, y_train_transformed, y_train, prefix="train_")

        self.is_fitted = True

        # Combine all metrics
        metrics = {**train_metrics, **val_metrics, **cv_scores, "feature_importance": self.feature_importance}

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict claim amounts"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Inverse transform if log was applied
        if self.log_transform:
            predictions = np.expm1(predictions)
            # Ensure non-negative predictions
            predictions = np.maximum(predictions, 0)

        return predictions

    def predict_with_uncertainty(self, X: pd.DataFrame, n_estimations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict claim amounts with uncertainty estimates
        Only works for tree-based models
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        if self.model_type not in ["random_forest", "gradient_boosting"]:
            # For non-tree models, return predictions with zero uncertainty
            predictions = self.predict(X)
            return predictions, np.zeros_like(predictions)

        X_scaled = self.scaler.transform(X)

        # For Random Forest, we can use the predictions from individual trees
        if self.model_type == "random_forest":
            tree_predictions = []
            for tree in self.model.estimators_:
                tree_pred = tree.predict(X_scaled)
                if self.log_transform:
                    tree_pred = np.expm1(tree_pred)
                tree_predictions.append(tree_pred)

            tree_predictions = np.array(tree_predictions)
            predictions = np.mean(tree_predictions, axis=0)
            uncertainties = np.std(tree_predictions, axis=0)
        else:
            # For other models, use the point prediction
            predictions = self.predict(X)
            uncertainties = np.zeros_like(predictions)

        return predictions, uncertainties

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info("Performing cross-validation...")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Use negative MSE for scoring (sklearn convention)
        cv_scores = cross_val_score(self.model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)

        # Convert back to positive MSE and calculate RMSE
        cv_mse = -cv_scores
        cv_rmse = np.sqrt(cv_mse)

        return {"cv_rmse_mean": cv_rmse.mean(), "cv_rmse_std": cv_rmse.std(), "cv_rmse_scores": cv_rmse.tolist()}

    def _evaluate(
        self, X: np.ndarray, y_transformed: np.ndarray, y_original: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        # Make predictions
        y_pred_transformed = self.model.predict(X)

        # Calculate metrics on transformed scale
        mse_transformed = mean_squared_error(y_transformed, y_pred_transformed)
        rmse_transformed = np.sqrt(mse_transformed)
        mae_transformed = mean_absolute_error(y_transformed, y_pred_transformed)
        r2_transformed = r2_score(y_transformed, y_pred_transformed)

        # Transform predictions back to original scale
        if self.log_transform:
            y_pred_original = np.expm1(y_pred_transformed)
            y_pred_original = np.maximum(y_pred_original, 0)
        else:
            y_pred_original = y_pred_transformed

        # Calculate metrics on original scale
        mse_original = mean_squared_error(y_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(y_original, y_pred_original)
        r2_original = r2_score(y_original, y_pred_original)

        # Calculate percentage errors
        mask = y_original > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_original[mask] - y_pred_original[mask]) / y_original[mask])) * 100
        else:
            mape = np.nan

        return {
            f"{prefix}rmse": rmse_original,
            f"{prefix}mae": mae_original,
            f"{prefix}r2": r2_original,
            f"{prefix}mape": mape,
            f"{prefix}rmse_transformed": rmse_transformed,
            f"{prefix}mae_transformed": mae_transformed,
            f"{prefix}r2_transformed": r2_transformed,
        }

    def _extract_feature_importance(self, feature_names: pd.Index):
        """Extract feature importance from the model"""
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_)
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
            "log_transform": getattr(self, "log_transform", True),
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
        self.log_transform = model_data.get("log_transform", True)
        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "training_metadata": self.training_metadata,
            "model_params": self.model.get_params() if self.model else {},
            "log_transform": getattr(self, "log_transform", True),
        }
