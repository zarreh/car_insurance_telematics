"""Telematics data preprocessing pipeline."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class TripSummary:
    """Processed trip data ready for modeling."""

    trip_id: str
    driver_id: str
    duration_minutes: float
    distance_miles: float
    average_speed_mph: float
    max_speed_mph: float
    harsh_braking_events: int
    harsh_acceleration_events: int
    sharp_cornering_events: int
    phone_usage_seconds: int
    speeding_percent: float
    night_driving: int
    rush_hour: int
    time_of_day: str
    start_zone: str
    end_zone: str


class EventDetector:
    """Detects driving events from sensor data."""

    def __init__(self, config: ProcessingConfig = DEFAULT_CONFIG):
        self.config = config
        self.g_force = 9.81  # m/s²

    def detect_harsh_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect harsh driving events from accelerometer data."""
        thresholds = self.config.sensor_thresholds

        df["harsh_brake"] = (df["accelerometer_x"] < -thresholds.harsh_braking_g * self.g_force).astype(int)

        df["harsh_accel"] = (df["accelerometer_x"] > thresholds.harsh_acceleration_g * self.g_force).astype(int)

        df["sharp_corner"] = (np.abs(df["accelerometer_y"]) > thresholds.sharp_cornering_g * self.g_force).astype(int)

        return df

    def detect_phone_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect phone usage during driving."""
        df["phone_use"] = (
            (df["screen_on"] == 1) | (df["app_foreground"].str.contains("phone|message|whatsapp", case=False, na=False))
        ).astype(int)

        # Filter out brief screen activations
        min_duration = self.config.sensor_thresholds.phone_use_min_duration_sec
        df["phone_use"] = df["phone_use"].rolling(min_duration, min_periods=1).sum() >= min_duration

        return df

    def detect_speeding(self, df: pd.DataFrame, speed_limit: float = 65.0) -> pd.DataFrame:
        """Detect speeding events."""
        buffer = self.config.sensor_thresholds.speeding_buffer_mph
        df["speeding"] = (df["speed_mph"] > speed_limit + buffer).astype(int)
        return df


class DataQualityValidator:
    """Validates data quality and completeness."""

    def __init__(self, config: ProcessingConfig = DEFAULT_CONFIG):
        self.config = config

    def validate_trip(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate trip data quality."""
        issues = []
        thresholds = self.config.quality_thresholds

        # GPS accuracy check
        avg_gps_accuracy = df["gps_accuracy_meters"].mean()
        if avg_gps_accuracy > thresholds.min_gps_accuracy_meters:
            issues.append(f"Poor GPS accuracy: {avg_gps_accuracy:.1f}m")

        # Data completeness check
        expected_readings = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        completeness = len(df) / expected_readings if expected_readings > 0 else 0
        if completeness < thresholds.min_data_completeness:
            issues.append(f"Incomplete data: {completeness:.1%}")

        # Trip duration check
        duration_minutes = expected_readings / 60
        if duration_minutes < thresholds.min_trip_duration_minutes:
            issues.append(f"Trip too short: {duration_minutes:.1f} minutes")

        # Speed sanity check
        max_speed = df["speed_mph"].max()
        if max_speed > thresholds.max_speed_mph:
            issues.append(f"Unrealistic speed: {max_speed:.1f} mph")

        is_valid = len(issues) == 0
        return is_valid, issues


class PrivacyTransformer:
    """Applies privacy-preserving transformations."""

    def __init__(self, config: ProcessingConfig = DEFAULT_CONFIG):
        self.config = config

    def hash_identifier(self, identifier: str) -> str:
        """Create anonymized hash of identifier."""
        salted = f"{identifier}{self.config.hash_salt}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]

    def create_location_zone(self, lat: float, lon: float) -> str:
        """Convert GPS coordinates to zone identifier."""
        precision = self.config.location_precision
        lat_rounded = round(lat, precision)
        lon_rounded = round(lon, precision)
        return f"zone_{lat_rounded}_{lon_rounded}"

    def get_time_category(self, timestamp: datetime) -> str:
        """Categorize time of day."""
        hour = timestamp.hour

        for category, (start, end) in self.config.feature_config.time_of_day_bins.items():
            if start <= end:
                if start <= hour < end:
                    return category
            else:  # Handle overnight periods
                if hour >= start or hour < end:
                    return category

        return "unknown"


class TripProcessor:
    """Main processor for converting raw sensor data to trip summaries."""

    def __init__(self, config: ProcessingConfig = DEFAULT_CONFIG):
        self.config = config
        self.event_detector = EventDetector(config)
        self.quality_validator = DataQualityValidator(config)
        self.privacy_transformer = PrivacyTransformer(config)

    def process_trip(self, df: pd.DataFrame) -> Optional[TripSummary]:
        """Process raw sensor data into trip summary."""
        try:
            # Validate data quality
            is_valid, issues = self.quality_validator.validate_trip(df)
            if not is_valid:
                logger.warning(f"Trip {df['trip_id'].iloc[0]} validation failed: {issues}")
                return None

            # Detect events
            df = self.event_detector.detect_harsh_events(df)
            df = self.event_detector.detect_phone_usage(df)
            df = self.event_detector.detect_speeding(df)

            # Calculate trip metrics
            trip_summary = self._calculate_trip_metrics(df)

            # Apply privacy transformations
            trip_summary = self._apply_privacy_transforms(trip_summary, df)

            return trip_summary

        except Exception as e:
            logger.error(f"Error processing trip {df['trip_id'].iloc[0]}: {str(e)}")
            return None

    def _calculate_trip_metrics(self, df: pd.DataFrame) -> TripSummary:
        """Calculate aggregated trip metrics."""
        start_time = df["timestamp"].min()
        end_time = df["timestamp"].max()

        # Time-based features
        duration_minutes = (end_time - start_time).total_seconds() / 60
        is_night = start_time.hour < 6 or start_time.hour > 20
        is_rush_hour = start_time.hour in [7, 8, 17, 18]

        # Distance calculation (simplified - in production use haversine)
        distance_miles = self._estimate_distance(df)

        return TripSummary(
            trip_id=df["trip_id"].iloc[0],
            driver_id=df["device_id"].iloc[0],
            duration_minutes=duration_minutes,
            distance_miles=distance_miles,
            average_speed_mph=df["speed_mph"].mean(),
            max_speed_mph=df["speed_mph"].max(),
            harsh_braking_events=df["harsh_brake"].sum(),
            harsh_acceleration_events=df["harsh_accel"].sum(),
            sharp_cornering_events=df["sharp_corner"].sum(),
            phone_usage_seconds=df["phone_use"].sum(),
            speeding_percent=df["speeding"].mean(),
            night_driving=int(is_night),
            rush_hour=int(is_rush_hour),
            time_of_day="",  # Will be set in privacy transforms
            start_zone="",  # Will be set in privacy transforms
            end_zone="",  # Will be set in privacy transforms
        )

    def _estimate_distance(self, df: pd.DataFrame) -> float:
        """Estimate trip distance from GPS coordinates."""
        # Simplified distance calculation
        # In production, use proper haversine formula
        lat_diff = df["gps_latitude"].max() - df["gps_latitude"].min()
        lon_diff = df["gps_longitude"].max() - df["gps_longitude"].min()

        # Very rough approximation (miles)
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 69.0
        return max(distance, 0.1)  # Minimum 0.1 miles

    def _apply_privacy_transforms(self, trip: TripSummary, df: pd.DataFrame) -> TripSummary:
        """Apply privacy-preserving transformations."""
        # Hash driver ID
        trip.driver_id = self.privacy_transformer.hash_identifier(trip.driver_id)

        # Convert locations to zones
        start_lat, start_lon = df.iloc[0][["gps_latitude", "gps_longitude"]]
        end_lat, end_lon = df.iloc[-1][["gps_latitude", "gps_longitude"]]

        trip.start_zone = self.privacy_transformer.create_location_zone(start_lat, start_lon)
        trip.end_zone = self.privacy_transformer.create_location_zone(end_lat, end_lon)

        # Categorize time
        trip.time_of_day = self.privacy_transformer.get_time_category(df["timestamp"].iloc[0])

        return trip


class BatchProcessor:
    """Processes multiple trips in batches."""

    def __init__(self, config: ProcessingConfig = DEFAULT_CONFIG):
        self.config = config
        self.trip_processor = TripProcessor(config)

    def process_batch(self, sensor_data: List[pd.DataFrame]) -> pd.DataFrame:
        """Process a batch of trips."""
        processed_trips = []

        for trip_data in sensor_data:
            trip_summary = self.trip_processor.process_trip(trip_data)
            if trip_summary:
                processed_trips.append(trip_summary)

        logger.info(f"Processed {len(processed_trips)}/{len(sensor_data)} trips successfully")

        # Convert to DataFrame
        if processed_trips:
            return pd.DataFrame([vars(trip) for trip in processed_trips])
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize processor
    processor = BatchProcessor()

    # Process trips (would load from files/database in production)
    logger.info("Telematics preprocessing pipeline initialized")
