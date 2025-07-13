"""
Run Inference Script
Main script to run inference using trained telematics ML models
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Add the modeling directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from car_insurance_telematics.modeling.inference_pipeline import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for testing inference"""
    sample_data = pd.DataFrame(
        {
            "driver_id": ["driver_001", "driver_002", "driver_003", "driver_004", "driver_005"],
            "trip_id": ["trip_001", "trip_002", "trip_003", "trip_004", "trip_005"],
            "trip_duration_minutes": [45.5, 30.2, 62.1, 15.8, 90.3],
            "trip_distance_km": [32.1, 18.5, 48.7, 8.2, 75.4],
            "max_speed_kmh": [95.0, 75.0, 125.0, 60.0, 110.0],
            "avg_speed_kmh": [65.2, 55.8, 78.9, 45.3, 82.1],
            "hard_braking_count": [2, 0, 4, 1, 3],
            "hard_acceleration_count": [1, 0, 3, 0, 2],
            "sharp_turn_count": [0, 1, 2, 0, 1],
            "night_driving_minutes": [10.0, 0.0, 45.0, 5.0, 30.0],
            "rush_hour_minutes": [20.0, 15.0, 0.0, 10.0, 45.0],
            "weekend_driving": [0, 1, 0, 1, 0],
            "phone_use_minutes": [5.0, 0.0, 8.0, 2.0, 0.0],
            "continuous_driving_hours": [0.75, 0.5, 1.0, 0.25, 1.5],
            "adverse_weather": [0, 0, 1, 0, 0],
            "rain_driving": [0, 0, 1, 0, 0],
            "high_traffic_ratio": [0.4, 0.2, 0.1, 0.6, 0.3],
        }
    )
    return sample_data


def main():
    """Main function to run inference"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on telematics data")
    parser.add_argument("--input-file", type=str, help="Path to input CSV file with trip data")
    parser.add_argument("--output-file", type=str, help="Path to save predictions (optional)")
    parser.add_argument(
        "--model-registry-dir", type=str, default="model_registry", help="Directory containing model registry"
    )
    parser.add_argument("--output-dir", type=str, default="data/ml_results", help="Directory to save inference results")
    parser.add_argument("--single-prediction", action="store_true", help="Make a single prediction (demo mode)")
    parser.add_argument("--use-sample-data", action="store_true", help="Use sample data for testing")
    parser.add_argument("--no-uncertainty", action="store_true", help="Disable uncertainty estimates")
    parser.add_argument("--export-production", action="store_true", help="Export models for production deployment")
    # # make argument for single prediction
    # parser.add_argument(
    #     "--model-type",
    #     type=str,
    #     default="random_forest",
    #     choices=["random_forest", "gradient_boosting", "logistic_regression"],
    #     help="Model type for claim probability",
    # )
    # parser.add_argument(
    #     "--severity-model-type",
    #     type=str,
    #     default="random_forest",
    #     choices=["random_forest", "gradient_boosting", "linear", "ridge", "lasso"],
    #     help="Model type for claim severity",
    # )

    args = parser.parse_args()

    # Initialize inference pipeline
    logger.info("Initializing InferencePipeline...")
    try:
        pipeline = InferencePipeline(model_registry_dir=args.model_registry_dir, output_dir=args.output_dir)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        logger.error("Make sure models have been trained first using train_models.py")
        return

    try:
        if args.export_production:
            # Export models for production
            logger.info("Exporting models for production deployment...")
            pipeline.export_for_production("production_models")
            print("" + "=" * 60)
            print("PRODUCTION EXPORT COMPLETED")
            print("=" * 60)
            print("Models exported to: production_models/")
            print("See DEPLOYMENT.md for deployment instructions")

        elif args.single_prediction:
            # Demo single prediction
            logger.info("Running single prediction demo...")

            sample_trip = {
                "driver_id": "demo_driver",
                "trip_id": "demo_trip",
                "trip_duration_minutes": 45.5,
                "trip_distance_km": 32.1,
                "max_speed_kmh": 95.0,
                "avg_speed_kmh": 65.2,
                "hard_braking_count": 2,
                "hard_acceleration_count": 1,
                "sharp_turn_count": 0,
                "night_driving_minutes": 10.0,
                "rush_hour_minutes": 20.0,
                "weekend_driving": 0,
                "phone_use_minutes": 5.0,
                "continuous_driving_hours": 0.75,
                "adverse_weather": 0,
                "rain_driving": 0,
                "high_traffic_ratio": 0.4,
            }

            result = pipeline.predict_single(sample_trip)

            print("" + "=" * 60)
            print("SINGLE PREDICTION RESULT")
            print("=" * 60)
            print(f"Driver ID: {result['driver_id']}")
            print(f"Trip ID: {result['trip_id']}")
            print("Predictions:")
            print("-" * 30)
            print(f"Claim Probability: {result['claim_probability']:.2%}")
            print(f"Risk Category: {result['risk_category']}")
            print(f"Risk Score: {result['risk_score']:.1f}/100")
            print(f"Expected Claim Amount: ${result['expected_claim_amount']:,.2f}")
            print(f"Severity Prediction: ${result['claim_severity_prediction']:,.2f}")

            if "severity_uncertainty" in result:
                print(f"Confidence Interval (95%):")
                print(f"  Lower: ${result['confidence_interval_lower']:,.2f}")
                print(f"  Upper: ${result['confidence_interval_upper']:,.2f}")

        else:
            # Batch prediction
            if args.use_sample_data:
                # Use sample data
                logger.info("Using sample data for inference...")
                data = create_sample_data()
                # Save sample data
                sample_file = os.path.join(args.output_dir, "sample_input_data.csv")
                data.to_csv(sample_file, index=False)
                input_file = sample_file
            else:
                # Use provided input file
                if not args.input_file:
                    logger.error("Please provide --input-file or use --use-sample-data")
                    return

                if not os.path.exists(args.input_file):
                    logger.error(f"Input file not found: {args.input_file}")
                    return

                input_file = args.input_file

            logger.info(f"Running batch inference on: {input_file}")

            # Run inference
            output_file = pipeline.predict_from_file(
                input_file=input_file, output_file=args.output_file, include_uncertainty=not args.no_uncertainty
            )

            # Load and display results summary
            results_df = pd.read_csv(output_file)

            print("" + "=" * 60)
            print("BATCH INFERENCE COMPLETED")
            print("=" * 60)
            print(f"Processed {len(results_df)} trips")
            print(f"Results saved to: {output_file}")

            # Summary statistics
            print("Risk Distribution:")
            print("-" * 30)
            risk_dist = results_df["risk_category"].value_counts()
            for category, count in risk_dist.items():
                print(f"{category:15s}: {count:4d} ({count/len(results_df)*100:5.1f}%)")

            print("Claim Probability Statistics:")
            print("-" * 30)
            print(f"Mean: {results_df['claim_probability'].mean():.3f}")
            print(f"Std:  {results_df['claim_probability'].std():.3f}")
            print(f"Min:  {results_df['claim_probability'].min():.3f}")
            print(f"Max:  {results_df['claim_probability'].max():.3f}")

            print("Expected Claim Amount Statistics:")
            print("-" * 30)
            print(f"Total Expected Claims: ${results_df['expected_claim_amount'].sum():,.2f}")
            print(f"Average per Trip: ${results_df['expected_claim_amount'].mean():,.2f}")
            print(f"Max Expected Claim: ${results_df['expected_claim_amount'].max():,.2f}")

            # High risk trips
            high_risk = results_df[results_df["claim_probability"] > 0.5]
            if len(high_risk) > 0:
                print(f"High Risk Trips ({len(high_risk)} trips):")
                print("-" * 30)
                print(high_risk[["driver_id", "trip_id", "claim_probability", "expected_claim_amount"]].head())

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

    logger.info("Inference script completed")


if __name__ == "__main__":
    main()
