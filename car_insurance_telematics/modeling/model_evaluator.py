"""
Model Evaluator
Provides evaluation metrics and visualization for telematics ML models
"""

import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and visualizes model performance"""

    def __init__(self, output_dir: str = "data/ml_results"):
        """
        Initialize the evaluator

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def evaluate_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate classification model performance

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            prefix: Prefix for metric names (e.g., 'train', 'test')

        Returns:
            Dictionary of evaluation metrics
        """
        # Handle probability array - get probabilities for positive class
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_positive = y_proba[:, 1]  # Get probabilities for class 1
        else:
            y_proba_positive = y_proba

        # Calculate metrics
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
            f"{prefix}_roc_auc": roc_auc_score(y_true, y_proba_positive) if len(np.unique(y_true)) > 1 else 0.0,
            f"{prefix}_pr_auc": (
                average_precision_score(y_true, y_proba_positive) if len(np.unique(y_true)) > 1 else 0.0
            ),
        }

        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update(
                {
                    f"{prefix}_true_negatives": int(tn),
                    f"{prefix}_false_positives": int(fp),
                    f"{prefix}_false_negatives": int(fn),
                    f"{prefix}_true_positives": int(tp),
                }
            )

        # Log results
        logger.info(f"{prefix.capitalize()} Classification Metrics:")
        logger.info(f"  Accuracy: {metrics[f'{prefix}_accuracy']:.4f}")
        logger.info(f"  Precision: {metrics[f'{prefix}_precision']:.4f}")
        logger.info(f"  Recall: {metrics[f'{prefix}_recall']:.4f}")
        logger.info(f"  F1 Score: {metrics[f'{prefix}_f1']:.4f}")
        logger.info(f"  ROC AUC: {metrics[f'{prefix}_roc_auc']:.4f}")
        logger.info(f"  PR AUC: {metrics[f'{prefix}_pr_auc']:.4f}")

        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Evaluate regression model performance

        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names

        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate metrics
        metrics = {
            f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}_r2": r2_score(y_true, y_pred),
            f"{prefix}_mape": mean_absolute_percentage_error(y_true, y_pred) if not np.any(y_true == 0) else np.nan,
        }

        # Calculate percentile errors
        errors = np.abs(y_true - y_pred)
        metrics.update(
            {
                f"{prefix}_median_error": np.median(errors),
                f"{prefix}_90th_percentile_error": np.percentile(errors, 90),
                f"{prefix}_95th_percentile_error": np.percentile(errors, 95),
            }
        )

        # Log results
        logger.info(f"{prefix.capitalize()} Regression Metrics:")
        logger.info(f"  RMSE: ${metrics[f'{prefix}_rmse']:.2f}")
        logger.info(f"  MAE: ${metrics[f'{prefix}_mae']:.2f}")
        logger.info(f"  R²: {metrics[f'{prefix}_r2']:.4f}")
        if not np.isnan(metrics[f"{prefix}_mape"]):
            logger.info(f"  MAPE: {metrics[f'{prefix}_mape']:.2%}")

        return metrics

    def plot_classification_results(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str = "model"
    ):
        """
        Create classification evaluation plots

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model_name: Name for saving plots
        """
        # Handle probability array
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        # 2. ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
            auc = roc_auc_score(y_true, y_proba_positive)
            axes[0, 1].plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
            axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random")
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("True Positive Rate")
            axes[0, 1].set_title("ROC Curve")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 3. Precision-Recall Curve
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
            pr_auc = average_precision_score(y_true, y_proba_positive)
            axes[1, 0].plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})")
            axes[1, 0].set_xlabel("Recall")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].set_title("Precision-Recall Curve")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # 4. Probability Distribution
        axes[1, 1].hist(y_proba_positive[y_true == 0], bins=30, alpha=0.5, label="No Claim", density=True)
        axes[1, 1].hist(y_proba_positive[y_true == 1], bins=30, alpha=0.5, label="Claim", density=True)
        axes[1, 1].set_xlabel("Predicted Probability")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Probability Distribution by Class")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{model_name}_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Classification plots saved to {plot_path}")

    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"):
        """
        Create regression evaluation plots

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name for saving plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        axes[0, 0].set_xlabel("Actual Claim Amount")
        axes[0, 0].set_ylabel("Predicted Claim Amount")
        axes[0, 0].set_title("Actual vs Predicted")
        axes[0, 0].grid(True)

        # 2. Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted Claim Amount")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residual Plot")
        axes[0, 1].grid(True)

        # 3. Residual Distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor="black")
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Residual Distribution")
        axes[1, 0].grid(True)

        # 4. Q-Q Plot
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{model_name}_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Regression plots saved to {plot_path}")

    def plot_feature_importance(self, feature_importance: pd.DataFrame, model_name: str = "model", top_n: int = 20):
        """
        Plot feature importance

        Args:
            feature_importance: DataFrame with feature names and importance scores
            model_name: Name for saving plot
            top_n: Number of top features to display
        """
        # Sort and get top features
        top_features = feature_importance.nlargest(top_n, "importance")

        plt.figure(figsize=(10, 8))
        plt.barh(top_features["feature"], top_features["importance"])
        plt.xlabel("Importance Score")
        plt.title(f"Top {top_n} Feature Importance - {model_name}")
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"{model_name}_feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to {plot_path}")

    def plot_model_comparison(self, comparison_results: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Plot model comparison results

        Args:
            comparison_results: Dictionary with model comparison metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Classification models comparison
        if comparison_results.get("claim_probability"):
            prob_df = pd.DataFrame(comparison_results["claim_probability"]).T
            prob_df.plot(kind="bar", ax=axes[0])
            axes[0].set_title("Claim Probability Model Comparison")
            axes[0].set_xlabel("Model Type")
            axes[0].set_ylabel("Score")
            axes[0].legend(loc="best")
            axes[0].grid(True)

        # Regression models comparison
        if comparison_results.get("claim_severity"):
            sev_df = pd.DataFrame(comparison_results["claim_severity"]).T
            # Normalize RMSE for better visualization
            if "test_rmse" in sev_df.columns:
                sev_df["test_rmse_normalized"] = 1 - (sev_df["test_rmse"] / sev_df["test_rmse"].max())
                sev_df = sev_df.drop("test_rmse", axis=1)
            sev_df.plot(kind="bar", ax=axes[1])
            axes[1].set_title("Claim Severity Model Comparison")
            axes[1].set_xlabel("Model Type")
            axes[1].set_ylabel("Score")
            axes[1].legend(loc="best")
            axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Model comparison plot saved to {plot_path}")

    def create_evaluation_report(
        self,
        model_name: str,
        model_type: str,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Create a comprehensive evaluation report

        Args:
            model_name: Name of the model
            model_type: Type of model
            train_metrics: Training metrics
            test_metrics: Test metrics
            feature_importance: Feature importance DataFrame

        Returns:
            Path to the report file
        """
        report = f"""
# Model Evaluation Report: {model_name}

## Model Information
- Model Type: {model_type}
- Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

### Training Set
"""
        for metric, value in train_metrics.items():
            if "train_" in metric:
                metric_name = metric.replace("train_", "").replace("_", " ").title()
                if isinstance(value, float):
                    report += f"- {metric_name}: {value:.4f}\n"
                else:
                    report += f"- {metric_name}: {value}\n"

        report += """
### Test Set
"""
        for metric, value in test_metrics.items():
            if "test_" in metric:
                metric_name = metric.replace("test_", "").replace("_", " ").title()
                if isinstance(value, float):
                    report += f"- {metric_name}: {value:.4f}\n"
                else:
                    report += f"- {metric_name}: {value}\n"

        if feature_importance is not None:
            report += """
## Top 10 Most Important Features
"""
            top_features = feature_importance.nlargest(10, "importance")
            for idx, row in top_features.iterrows():
                report += f"{idx + 1}. {row['feature']}: {row['importance']:.4f}\n"

        # Save report
        report_path = os.path.join(self.output_dir, f"{model_name}_evaluation_report.md")
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {report_path}")

        return report_path


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator(output_dir="data/ml_results")

    # Example evaluation
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples, 2)

    # Evaluate classification
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba, prefix="test")

    # Create plots
    evaluator.plot_classification_results(y_true, y_pred, y_proba, "example_model")
