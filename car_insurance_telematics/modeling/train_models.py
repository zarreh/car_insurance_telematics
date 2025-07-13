"""
Train Models Script
Main script to train telematics ML models
"""

import argparse
import logging
import os
import sys

# Add the modeling directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from car_insurance_telematics.modeling.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main function to train models"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train telematics ML models")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/processed_trips_1200_drivers.csv",
        help="Path to processed data CSV file",
    )
    parser.add_argument("--output-dir", type=str, default="data/ml_results", help="Directory to save training results")
    parser.add_argument("--model-registry-dir", type=str, default="model_registry", help="Directory for model registry")
    parser.add_argument(
        "--prob-model",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        help="Model type for claim probability",
    )
    parser.add_argument(
        "--severity-model",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "linear", "ridge", "lasso"],
        help="Model type for claim severity",
    )
    parser.add_argument("--compare-models", action="store_true", help="Compare multiple model types")

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return

    # Initialize trainer
    logger.info("Initializing ModelTrainer...")
    trainer = ModelTrainer(
        data_path=args.data_path, output_dir=args.output_dir, model_registry_dir=args.model_registry_dir
    )

    try:
        if args.compare_models:
            # Compare different model types
            logger.info("Comparing multiple model types...")
            comparison_results = trainer.compare_models(
                model_types=["random_forest", "gradient_boosting", "logistic_regression"]
            )

            # Print comparison results
            print("" + "=" * 60)
            print("MODEL COMPARISON RESULTS")
            print("=" * 60)

            print("Claim Probability Models:")
            print("-" * 30)
            for model_type, metrics in comparison_results["claim_probability"].items():
                print(f"{model_type}:")
                print(f"  ROC AUC: {metrics['test_auc']:.4f}")
                print(f"  PR AUC: {metrics['test_pr_auc']:.4f}")

            print("Claim Severity Models:")
            print("-" * 30)
            for model_type, metrics in comparison_results["claim_severity"].items():
                print(f"{model_type}:")
                print(f"  RMSE: ${metrics['test_rmse']:,.2f}")
                print(f"  R²: {metrics['test_r2']:.4f}")
                print(f"  MAE: ${metrics['test_mae']:,.2f}")

        else:
            # Train specified models
            logger.info(f"Training models: probability={args.prob_model}, severity={args.severity_model}")
            results = trainer.train_all_models(
                probability_model_type=args.prob_model, severity_model_type=args.severity_model
            )

            # Print training results
            print("" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 60)

            print("Claim Probability Model:")
            print("-" * 30)
            prob_metrics = results["probability_model"]["results"]["test_metrics"]
            print(f"Model Type: {args.prob_model}")
            print(f"Test ROC AUC: {prob_metrics['test_roc_auc']:.4f}")
            print(f"Test PR AUC: {prob_metrics['test_pr_auc']:.4f}")
            print(f"Test Accuracy: {prob_metrics['test_accuracy']:.4f}")
            print(f"Test F1 Score: {prob_metrics['test_f1']:.4f}")
            print("Claim Severity Model:")
            print("-" * 30)
            sev_metrics = results["severity_model"]["results"]["test_metrics"]
            print(f"Model Type: {args.severity_model}")
            print(f"Test RMSE: ${sev_metrics['test_rmse']:,.2f}")
            print(f"Test R²: {sev_metrics['test_r2']:.4f}")
            print(f"Test MAE: ${sev_metrics['test_mae']:,.2f}")
            if "test_mape" in sev_metrics and sev_metrics["test_mape"] is not None:
                print(f"Test MAPE: {sev_metrics['test_mape']:.2f}%")

            print("Top 10 Important Features (Claim Probability):")
            print("-" * 30)
            prob_importance = results["probability_model"]["results"]["feature_importance"]
            for idx, row in prob_importance.head(10).iterrows():
                print(f"{row['feature']:30s} {row['importance']:.4f}")

            print("Model Paths:")
            print("-" * 30)
            print(f"Probability Model: {results['summary']['models']['claim_probability']['model_path']}")
            print(f"Severity Model: {results['summary']['models']['claim_severity']['model_path']}")

            print(f"Full results saved to: {args.output_dir}")
            print(f"Model registry: {args.model_registry_dir}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    logger.info("Training script completed")


if __name__ == "__main__":
    main()
