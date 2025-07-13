"""Data loading utilities for telematics pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of raw sensor data from various sources."""

    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_json_files(self, pattern: str = "*.json") -> Generator[pd.DataFrame, None, None]:
        """Load JSON files matching pattern and yield as DataFrames."""
        json_files = list(self.data_dir.glob(pattern))
        logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")

        for file_path in json_files:
            try:
                dfs = self._load_single_json(file_path)
                if dfs:
                    for df in dfs:
                        if df is not None and not df.empty:
                            yield df
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")

    def _load_single_json(self, file_path: Path) -> List[Optional[pd.DataFrame]]:
        """Load a single JSON file containing sensor data."""
        with open(file_path, "r") as f:
            data = json.load(f)

        dfs = []

        # Handle different JSON structures
        if isinstance(data, list):
            # Could be list of trips or list of readings
            if data and isinstance(data[0], dict):
                if "sensor_readings" in data[0]:
                    # List of trips
                    for trip_data in data:
                        df = self._extract_sensor_readings(trip_data, file_path)
                        if df is not None:
                            dfs.append(df)
                else:
                    # Direct list of sensor readings
                    df = pd.DataFrame(data)
                    df = self._prepare_dataframe(df, file_path)
                    dfs.append(df)

        elif isinstance(data, dict):
            if "sensor_readings" in data:
                # Single trip with sensor_readings
                df = self._extract_sensor_readings(data, file_path)
                if df is not None:
                    dfs.append(df)
            elif "readings" in data:
                # Alternative structure with 'readings' key
                df = pd.DataFrame(data["readings"])
                df = self._prepare_dataframe(df, file_path)
                dfs.append(df)
            else:
                # Try to convert dict directly to DataFrame
                df = pd.DataFrame([data])
                df = self._prepare_dataframe(df, file_path)
                dfs.append(df)
        else:
            logger.warning(f"Unexpected data format in {file_path}")

        return dfs

    def _extract_sensor_readings(self, trip_data: dict, file_path: Path) -> Optional[pd.DataFrame]:
        """Extract sensor readings from trip data."""
        if "sensor_readings" not in trip_data:
            return None

        readings = trip_data["sensor_readings"]
        if not readings:
            return None

        df = pd.DataFrame(readings)

        # Add trip metadata if available
        if "trip_metadata" in trip_data:
            metadata = trip_data["trip_metadata"]
            for key, value in metadata.items():
                if key not in df.columns:
                    df[key] = value

        return self._prepare_dataframe(df, file_path)

    def _prepare_dataframe(self, df: pd.DataFrame, file_path: Path) -> Optional[pd.DataFrame]:
        """Prepare DataFrame with required fields and conversions."""
        if df.empty:
            return None

        # Ensure required fields exist with defaults
        required_fields = {
            "timestamp": None,
            "gps_latitude": 0.0,
            "gps_longitude": 0.0,
            "gps_accuracy_meters": 20.0,
            "speed_mph": 0.0,
            "accelerometer_x": 0.0,
            "accelerometer_y": 0.0,
            "accelerometer_z": 9.81,
            "screen_on": 0,
            "app_foreground": "unknown",
        }

        for field, default in required_fields.items():
            if field not in df.columns:
                logger.debug(f"Adding missing field '{field}' with default value to {file_path.name}")
                df[field] = default

        # Convert timestamp strings to datetime
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except:
                logger.warning(f"Could not parse timestamps in {file_path.name}")
                return None
        else:
            # Generate timestamps if missing
            logger.debug(f"Generating timestamps for {file_path.name}")
            df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="S")

        # Add trip_id if not present
        if "trip_id" not in df.columns:
            df["trip_id"] = file_path.stem

        # Add device_id if not present
        if "device_id" not in df.columns:
            df["device_id"] = df.get("device_id", "unknown_device")

        # Ensure numeric fields are numeric
        numeric_fields = [
            "gps_latitude",
            "gps_longitude",
            "gps_accuracy_meters",
            "speed_mph",
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
        ]
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0)

        logger.debug(f"Loaded {len(df)} records from {file_path.name}")
        return df

    def load_batch(self, batch_size: int = 100) -> Generator[List[pd.DataFrame], None, None]:
        """Load a batch of trips for processing."""
        batch = []
        for df in self.load_json_files():
            batch.append(df)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining trips
        if batch:
            yield batch


class DataSaver:
    """Handles saving of processed data."""

    def __init__(self, output_dir: Path = Path("data/processed")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_trips(self, df: pd.DataFrame, filename: str = None) -> Path:
        """Save processed trips to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_trips_{timestamp}.csv"

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} trips to {output_path}")
        return output_path

    def save_driver_aggregates(self, df: pd.DataFrame, filename: str = "driver_features.csv") -> Path:
        """Save driver-level aggregated features."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} driver profiles to {output_path}")
        return output_path
