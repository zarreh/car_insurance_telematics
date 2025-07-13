"""
Claim Probability Model
Binary classification model to predict likelihood of insurance claims
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimProbabilityModel:
    """Model for predicting claim probability"""

    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the claim probability model

        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.calibrated_model = None

    def _create_model(self, early_stopping_rounds: Optional[int] = None) -> xgb.XGBClassifier:
        """Create the base model"""
        if self.model_type == "xgboost":
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': 42,
                'scale_pos_weight': 1,  # Will be adjusted based on class imbalance
                'tree_method': 'hist'
            }

            # Add early_stopping_rounds to constructor if provided
            if early_stopping_rounds is not None:
                params['early_stopping_rounds'] = early_stopping_rounds
                params['n_estimators'] = 1000  # Set higher when using early stopping

            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train the claim probability model

        Args:
            X: Feature matrix
            y: Target labels (0/1)
            validation_data: Optional tuple of (X_val, y_val)

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle class imbalance
        pos_weight = len(y[y == 0]) / len(y[y == 1])

        # Create and configure model
        early_stopping = 10 if validation_data is not None else None
        self.model = self._create_model(early_stopping_rounds=early_stopping)
        self.model.scale_pos_weight = pos_weight

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            self._create_model(),  # Use fresh model for CV
            X_scaled, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )

        # Train the model
        fit_params = {}
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            fit_params['eval_set'] = [(X_val_scaled, y_val)]
            fit_params['verbose'] = False

        self.model.fit(X_scaled, y, **fit_params)

        # Calibrate probabilities
        logger.info("Calibrating model probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='sigmoid', cv=3
        )
        self.calibrated_model.fit(X_scaled, y)

        self.is_fitted = True

        # Calculate training metrics
        train_pred_proba = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        train_auc = roc_auc_score(y, train_pred_proba)

        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(y, train_pred_proba)
        pr_auc = auc(recall, precision)

        metrics = {
            'model_type': self.model_type,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'train_auc': train_auc,
            'train_pr_auc': pr_auc,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'positive_rate': y.mean()
        }

        logger.info(f"Training completed. CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0/1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.calibrated_model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict claim probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability array with shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.calibrated_model.predict_proba(X_scaled)

    @property
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before accessing feature importance")

        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        else:
            # For models without feature_importances_, return uniform importance
            importance_scores = np.ones(len(self.feature_names)) / len(self.feature_names)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, filepath: str):
        """Save the model to disk"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the model from disk"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.calibrated_model = model_data['calibrated_model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Model loaded from {filepath}")

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.is_fitted:
            return {}

        params = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
        }

        if hasattr(self.model, 'get_params'):
            params.update(self.model.get_params())

        return params
