"""
Model Evaluator
Comprehensive model evaluation and visualization utilities
"""

import logging
import os
import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_recall_curve,
                             r2_score, roc_curve)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""

    def __init__(self, output_dir: str = "data/ml_results"):
        """
        Initialize the model evaluator

        Args:
            output_dir: Directory to save evaluation results and plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def evaluate_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model_name: Name of the model for saving results

        Returns:
            Dictionary containing evaluation metrics and plot paths
        """
        logger.info(f"Evaluating classification model: {model_name}")

        # Calculate metrics
        metrics = self._calculate_classification_metrics(y_true, y_pred, y_proba)

        # Generate plots
        plot_paths = self._generate_classification_plots(y_true, y_pred, y_proba, model_name)

        # Generate report
        report = self._generate_classification_report(y_true, y_pred, y_proba, metrics, model_name)

        return {"metrics": metrics, "plots": plot_paths, "report_path": report}

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation for regression models

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for saving results

        Returns:
            Dictionary containing evaluation metrics and plot paths
        """
        logger.info(f"Evaluating regression model: {model_name}")

        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_true, y_pred)

        # Generate plots
        plot_paths = self._generate_regression_plots(y_true, y_pred, model_name)

        # Generate report
        report = self._generate_regression_report(y_true, y_pred, metrics, model_name)

        return {"metrics": metrics, "plots": plot_paths, "report_path": report}

    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate various metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        # PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        avg_precision = average_precision_score(y_true, y_proba)

        # Brier score (calibration metric)
        brier_score = np.mean((y_proba - y_true) ** 2)

        return {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
            "test_avg_precision": avg_precision,
            "test_brier_score": brier_score,
            "test_true_negatives": int(tn),
            "test_false_positives": int(fp),
            "test_false_negatives": int(fn),
            "test_true_positives": int(tp),
        }

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate MAPE only for non-zero values
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = np.nan

        # Calculate percentile errors
        errors = np.abs(y_true - y_pred)
        p50_error = np.percentile(errors, 50)
        p90_error = np.percentile(errors, 90)
        p95_error = np.percentile(errors, 95)

        return {
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "test_median_error": p50_error,
            "test_p90_error": p90_error,
            "test_p95_error": p95_error,
        }

    def _generate_classification_plots(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str
    ) -> Dict[str, str]:
        """Generate classification evaluation plots"""
        plot_paths = {}

        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["confusion_matrix"] = cm_path

        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        roc_path = os.path.join(self.output_dir, f"{model_name}_roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["roc_curve"] = roc_path

        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.legend(loc="lower left")
        pr_path = os.path.join(self.output_dir, f"{model_name}_pr_curve.png")
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["pr_curve"] = pr_path

        # 4. Calibration Plot
        plt.figure(figsize=(8, 6))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot - {model_name}")
        plt.legend(loc="lower right")
        cal_path = os.path.join(self.output_dir, f"{model_name}_calibration.png")
        plt.savefig(cal_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["calibration"] = cal_path

        # 5. Probability Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba[y_true == 0], bins=50, alpha=0.5, label="No Claim", density=True)
        plt.hist(y_proba[y_true == 1], bins=50, alpha=0.5, label="Claim", density=True)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title(f"Probability Distribution by Class - {model_name}")
        plt.legend()
        prob_dist_path = os.path.join(self.output_dir, f"{model_name}_prob_distribution.png")
        plt.savefig(prob_dist_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["prob_distribution"] = prob_dist_path

        return plot_paths

    def _generate_regression_plots(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, str]:
        """Generate regression evaluation plots"""
        plot_paths = {}

        # 1. Actual vs Predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted - {model_name}")
        actual_pred_path = os.path.join(self.output_dir, f"{model_name}_actual_vs_predicted.png")
        plt.savefig(actual_pred_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["actual_vs_predicted"] = actual_pred_path

        # 2. Residual Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot - {model_name}")
        residual_path = os.path.join(self.output_dir, f"{model_name}_residuals.png")
        plt.savefig(residual_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["residuals"] = residual_path

        # 3. Residual Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor="black")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title(f"Residual Distribution - {model_name}")
        plt.axvline(x=0, color="r", linestyle="--")
        residual_dist_path = os.path.join(self.output_dir, f"{model_name}_residual_distribution.png")
        plt.savefig(residual_dist_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["residual_distribution"] = residual_dist_path

        # 4. Q-Q Plot
        from scipy import stats

        plt.figure(figsize=(8, 8))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot - {model_name}")
        qq_path = os.path.join(self.output_dir, f"{model_name}_qq_plot.png")
        plt.savefig(qq_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["qq_plot"] = qq_path

        # 5. Error Distribution by Magnitude
        plt.figure(figsize=(10, 6))
        relative_errors = np.abs(residuals) / (np.abs(y_true) + 1e-10)
        plt.scatter(y_true, relative_errors, alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Relative Error")
        plt.title(f"Relative Error vs Actual Values - {model_name}")
        plt.yscale("log")
        error_dist_path = os.path.join(self.output_dir, f"{model_name}_error_distribution.png")
        plt.savefig(error_dist_path, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths["error_distribution"] = error_dist_path

        return plot_paths

    def _generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, metrics: Dict[str, float], model_name: str
    ) -> str:
        """Generate detailed classification report"""
        report_path = os.path.join(self.output_dir, f"{model_name}_evaluation_report.txt")

        with open(report_path, "w") as f:
            f.write(f"Classification Model Evaluation Report: {model_name}\n")
            f.write("=" * 60 + "\n\n")

            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")

            # Classification report
            f.write("\n\nDetailed Classification Report:\n")
            f.write("-" * 30 + "\n")
            f.write(classification_report(y_true, y_pred, target_names=["No Claim", "Claim"]))

            # Threshold analysis
            f.write("\n\nThreshold Analysis:\n")
            f.write("-" * 30 + "\n")
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            for threshold in thresholds:
                y_pred_thresh = (y_proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f.write(f"\nThreshold: {threshold}\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall: {recall:.4f}\n")
                f.write(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")

        return report_path

    def _generate_regression_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float], model_name: str
    ) -> str:
        """Generate detailed regression report"""
        report_path = os.path.join(self.output_dir, f"{model_name}_evaluation_report.txt")

        residuals = y_true - y_pred

        with open(report_path, "w") as f:
            f.write(f"Regression Model Evaluation Report: {model_name}\n")
            f.write("=" * 60 + "\n\n")

            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")

            # Residual analysis
            f.write("\n\nResidual Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean Residual: {np.mean(residuals):.4f}\n")
            f.write(f"Std Residual: {np.std(residuals):.4f}\n")
            f.write(f"Min Residual: {np.min(residuals):.4f}\n")
            f.write(f"Max Residual: {np.max(residuals):.4f}\n")

            # Error percentiles
            f.write("\n\nError Percentiles:\n")
            f.write("-" * 30 + "\n")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            errors = np.abs(residuals)
            for p in percentiles:
                f.write(f"{p}th percentile: {np.percentile(errors, p):.4f}\n")

            # Prediction range analysis
            f.write("\n\nPrediction Range Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Actual range: [{y_true.min():.2f}, {y_true.max():.2f}]\n")
            f.write(f"Predicted range: [{y_pred.min():.2f}, {y_pred.max():.2f}]\n")
            f.write(f"Actual mean: {y_true.mean():.2f}\n")
            f.write(f"Predicted mean: {y_pred.mean():.2f}\n")
            f.write(f"Actual std: {y_true.std():.2f}\n")
            f.write(f"Predicted std: {y_pred.std():.2f}\n")

        return report_path

    def compare_models(self, results_dict: Dict[str, Dict[str, Any]], model_type: str = "classification") -> str:
        """
        Compare multiple models and generate comparison report

        Args:
            results_dict: Dictionary of model results
            model_type: Type of models ('classification' or 'regression')

        Returns:
            Path to comparison report
        """
        comparison_path = os.path.join(self.output_dir, f"{model_type}_model_comparison.png")

        # Extract metrics for comparison
        model_names = list(results_dict.keys())

        if model_type == "classification":
            metrics_to_compare = ["test_roc_auc", "test_pr_auc", "test_f1", "test_accuracy"]
        else:
            metrics_to_compare = ["test_r2", "test_rmse", "test_mae", "test_mape"]

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics_to_compare):
            values = [results_dict[model]["metrics"].get(metric, 0) for model in model_names]

            axes[idx].bar(model_names, values)
            axes[idx].set_title(metric.replace("test_", "").upper())
            axes[idx].set_ylabel("Score")
            axes[idx].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()

        return comparison_path
