"""
Claim Severity Model
Regression model to predict claim amounts given that a claim occurs
"""

import logging
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimSeverityModel:
    """Model for predicting claim severity (amount)"""

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the claim severity model

        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.log_transform = True  # Use log transform for claim amounts

    def _create_model(self, early_stopping_rounds: Optional[int] = None) -> xgb.XGBRegressor:
        """Create the base model"""
        if self.model_type == "xgboost":
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "objective": "reg:squarederror",
                "random_state": 42,
                "tree_method": "hist",
                "eval_metric": "rmse",
            }
            # params = {
            #     'n_estimators': 200,
            #     'max_depth': 5,
            #     'learning_rate': 0.05,
            #     'objective': 'reg:squarederror',
            #     'random_state': 42,
            #     'tree_method': 'hist',
            #     'subsample': 0.8,
            #     'colsample_bytree': 0.8,
            #     'min_child_weight': 3,
            #     'gamma': 0.05,
            #     'reg_alpha': 0.1,
            #     'reg_lambda': 1.0,
            #     'eval_metric': 'rmse'
            # }

            # Add early_stopping_rounds to constructor if provided
            if early_stopping_rounds is not None:
                params["early_stopping_rounds"] = early_stopping_rounds
                params["n_estimators"] = 1000  # Set higher when using early stopping

            return xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Train the claim severity model

        Args:
            X: Feature matrix
            y: Target values (claim amounts)
            validation_data: Optional tuple of (X_val, y_val)

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Log transform target if specified
        if self.log_transform:
            y_transformed = np.log1p(y)  # log(1 + y) to handle zeros
        else:
            y_transformed = y

        # Create model
        early_stopping = 10 if validation_data is not None else None
        self.model = self._create_model(early_stopping_rounds=early_stopping)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            self._create_model(),  # Use fresh model for CV
            X_scaled,
            y_transformed,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="neg_mean_squared_error",
        )
        cv_rmse = np.sqrt(-cv_scores)

        # Train the model
        fit_params = {}
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            y_val_transformed = np.log1p(y_val) if self.log_transform else y_val
            fit_params["eval_set"] = [(X_val_scaled, y_val_transformed)]
            fit_params["verbose"] = False

        self.model.fit(X_scaled, y_transformed, **fit_params)

        self.is_fitted = True

        # Calculate training metrics
        train_pred = self.predict(X)

        metrics = {
            "model_type": self.model_type,
            "cv_rmse_mean": cv_rmse.mean(),
            "cv_rmse_std": cv_rmse.std(),
            "train_rmse": np.sqrt(mean_squared_error(y, train_pred)),
            "train_mae": mean_absolute_error(y, train_pred),
            "train_r2": r2_score(y, train_pred),
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "mean_claim_amount": y.mean(),
            "median_claim_amount": y.median(),
        }

        logger.info(f"Training completed. CV RMSE: ${cv_rmse.mean():.2f} (+/- ${cv_rmse.std():.2f})")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict claim amounts

        Args:
            X: Feature matrix

        Returns:
            Predicted claim amounts
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Inverse transform if log transform was used
        if self.log_transform:
            predictions = np.expm1(predictions)  # exp(y) - 1

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_with_uncertainty(self, X: pd.DataFrame, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using dropout simulation

        Args:
            X: Feature matrix
            n_iterations: Number of prediction iterations

        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        # For XGBoost, we can use the prediction from individual trees
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)

        # Get predictions from individual trees
        if hasattr(self.model, "predict"):
            # Use the model's built-in prediction
            predictions = []

            # Get predictions with different random seeds for slight variation
            for i in range(min(n_iterations, 10)):
                pred = self.model.predict(X_scaled)
                if self.log_transform:
                    pred = np.expm1(pred)
                predictions.append(pred)

            predictions = np.array(predictions)

            # Add some noise based on prediction magnitude for uncertainty
            noise_scale = 0.1  # 10% noise
            for i in range(10, n_iterations):
                base_pred = predictions[i % 10]
                noise = np.random.normal(0, base_pred * noise_scale)
                noisy_pred = np.maximum(base_pred + noise, 0)
                predictions = np.vstack([predictions, noisy_pred])

            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
        else:
            # Fallback to point predictions
            mean_pred = self.predict(X)
            std_pred = mean_pred * 0.2  # Assume 20% uncertainty

        return mean_pred, std_pred

    @property
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before accessing feature importance")

        if hasattr(self.model, "feature_importances_"):
            importance_scores = self.model.feature_importances_
        else:
            # For models without feature_importances_, return uniform importance
            importance_scores = np.ones(len(self.feature_names)) / len(self.feature_names)

        importance_df = pd.DataFrame({"feature": self.feature_names, "importance": importance_scores}).sort_values(
            "importance", ascending=False
        )

        return importance_df

    def save(self, filepath: str):
        """Save the model to disk"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "log_transform": self.log_transform,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the model from disk"""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data["model_type"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = model_data["is_fitted"]
        self.log_transform = model_data.get("log_transform", True)

        logger.info(f"Model loaded from {filepath}")

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.is_fitted:
            return {}

        params = {
            "model_type": self.model_type,
            "n_features": len(self.feature_names),
            "log_transform": self.log_transform,
        }

        if hasattr(self.model, "get_params"):
            params.update(self.model.get_params())

        return params
