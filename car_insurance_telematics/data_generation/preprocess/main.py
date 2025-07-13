"""Main entry point for telematics data processing pipeline."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from config import ProcessingConfig
from data_loader import DataLoader, DataSaver
from feature_engineering import DriverAggregator
from preprocess import BatchProcessor


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/processing.log"), logging.StreamHandler()],
    )


def main(args):
    """Main processing pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting telematics processing pipeline")

    # Initialize components
    config = ProcessingConfig()
    data_loader = DataLoader(Path(args.input_dir))
    data_saver = DataSaver(Path(args.output_dir))
    batch_processor = BatchProcessor(config)

    # Process all trips
    all_processed_trips = []

    for batch in data_loader.load_batch(batch_size=args.batch_size):
        logger.info(f"Processing batch of {len(batch)} trips")

        # Process batch
        processed_df = batch_processor.process_batch(batch)

        if not processed_df.empty:
            all_processed_trips.append(processed_df)

            # Save intermediate results if specified
            if args.save_intermediate:
                data_saver.save_trips(processed_df)

    # Combine all processed trips
    if all_processed_trips:
        all_trips_df = pd.concat(all_processed_trips, ignore_index=True)

        # Save final trip-level data
        data_saver.save_trips(all_trips_df, "all_processed_trips.csv")

        # Aggregate to driver level if requested
        if args.aggregate_drivers:
            logger.info("Aggregating to driver level")
            aggregator = DriverAggregator()
            driver_features = aggregator.aggregate_trips(all_trips_df)
            data_saver.save_driver_aggregates(driver_features)

        logger.info(f"Pipeline completed successfully. Processed {len(all_trips_df)} trips")
    else:
        logger.warning("No trips were successfully processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process telematics sensor data")
    parser.add_argument("--input-dir", default="data/raw", help="Directory containing raw JSON files")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for processed CSV files")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of trips to process in each batch")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate batch results")
    parser.add_argument("--aggregate-drivers", action="store_true", help="Aggregate trips to driver level")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Setup logging
    setup_logging(args.log_level)

    # Run pipeline
    main(args)
