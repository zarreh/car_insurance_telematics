"""Configuration settings for telematics data processing pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SensorThresholds:
    """Thresholds for detecting driving events from sensor data."""

    harsh_braking_g: float = 0.3
    harsh_acceleration_g: float = 0.3
    sharp_cornering_g: float = 0.3
    speeding_buffer_mph: float = 5.0
    phone_use_min_duration_sec: int = 3


@dataclass
class DataQualityThresholds:
    """Minimum requirements for data quality."""

    min_gps_accuracy_meters: float = 50.0
    min_data_completeness: float = 0.8
    min_trip_duration_minutes: float = 1.0
    max_speed_mph: float = 150.0


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    speed_bins: List[float] = None
    time_of_day_bins: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        if self.speed_bins is None:
            self.speed_bins = [0, 25, 45, 65, 85, 150]

        if self.time_of_day_bins is None:
            self.time_of_day_bins = {
                "early_morning": (5, 7),
                "morning_commute": (7, 9),
                "midday": (9, 16),
                "evening_commute": (16, 19),
                "evening": (19, 22),
                "night": (22, 5),
            }


@dataclass
class ProcessingConfig:
    """Main configuration for the processing pipeline."""

    sensor_thresholds: SensorThresholds = None
    quality_thresholds: DataQualityThresholds = None
    feature_config: FeatureConfig = None

    # Processing settings
    batch_size: int = 1000
    parallel_workers: int = 4

    # Privacy settings
    hash_salt: str = "telematics_2025"
    location_precision: int = 2  # decimal places for GPS rounding

    def __post_init__(self):
        if self.sensor_thresholds is None:
            self.sensor_thresholds = SensorThresholds()
        if self.quality_thresholds is None:
            self.quality_thresholds = DataQualityThresholds()
        if self.feature_config is None:
            self.feature_config = FeatureConfig()


# Default configuration instance
DEFAULT_CONFIG = ProcessingConfig()
