"""Feature engineering for driver-level aggregation."""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class DriverAggregator:
    """Aggregates trip-level data to driver-level features."""

    def __init__(self):
        self.aggregation_rules = self._define_aggregation_rules()

    def _define_aggregation_rules(self) -> Dict[str, List[str]]:
        """Define how to aggregate each feature."""
        return {
            "trip_id": ["count"],
            "duration_minutes": ["sum", "mean", "std"],
            "distance_miles": ["sum", "mean", "std"],
            "average_speed_mph": ["mean", "std"],
            "max_speed_mph": ["mean", "max"],
            "harsh_braking_events": ["sum", "mean"],
            "harsh_acceleration_events": ["sum", "mean"],
            "sharp_cornering_events": ["sum", "mean"],
            "phone_usage_seconds": ["sum", "mean"],
            "speeding_percent": ["mean", "max"],
            "night_driving": ["mean"],
            "rush_hour": ["mean"],
        }

    def aggregate_trips(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trip data to driver level."""
        logger.info(f"Aggregating {len(trips_df)} trips for {trips_df['driver_id'].nunique()} drivers")

        # Basic aggregation
        driver_features = trips_df.groupby("driver_id").agg(self.aggregation_rules)

        # Flatten column names
        driver_features.columns = ["_".join(col).strip() for col in driver_features.columns.values]
        driver_features = driver_features.reset_index()

        # Rename trip count column
        driver_features.rename(columns={"trip_id_count": "total_trips"}, inplace=True)

        # Add derived features
        driver_features = self._add_derived_features(driver_features)

        # Add risk indicators
        driver_features = self._add_risk_indicators(driver_features, trips_df)

        return driver_features

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features."""
        # Events per mile
        df["events_per_mile"] = (
            df["harsh_braking_events_sum"] + df["harsh_acceleration_events_sum"] + df["sharp_cornering_events_sum"]
        ) / df["distance_miles_sum"].clip(lower=1)

        # Phone usage per hour
        df["phone_usage_per_hour"] = df["phone_usage_seconds_sum"] / (df["duration_minutes_sum"] / 60).clip(lower=0.1)

        # Consistency score (lower std is better)
        df["speed_consistency"] = 1 / (1 + df.get("average_speed_mph_std", 0))

        return df

    def _add_risk_indicators(self, df: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-related features."""
        # Time of day distribution
        time_dist = trips_df.groupby(["driver_id", "time_of_day"]).size().unstack(fill_value=0)
        time_dist_pct = time_dist.div(time_dist.sum(axis=1), axis=0)

        # Add high-risk time percentages
        high_risk_times = ["night", "early_morning"]
        for time_cat in high_risk_times:
            if time_cat in time_dist_pct.columns:
                df = df.merge(
                    time_dist_pct[time_cat].rename(f"{time_cat}_driving_pct"),
                    left_on="driver_id",
                    right_index=True,
                    how="left",
                )

        return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    aggregator = DriverAggregator()

    # Simulated trip data
    trips_data = {
        "driver_id": [1, 1, 2, 2],
        "trip_id": [101, 102, 201, 202],
        "duration_minutes": [30, 45, 20, 25],
        "distance_miles": [10, 15, 5, 7],
        "average_speed_mph": [20, 25, 15, 14],
        "max_speed_mph": [30, 35, 20, 22],
        "harsh_braking_events": [1, 2, 0, 1],
        "harsh_acceleration_events": [0, 1, 1, 0],
        "sharp_cornering_events": [1, 0, 0, 1],
        "phone_usage_seconds": [120, 180, 60, 90],
        "speeding_percent": [10.0, 15.0, 5.0, 8.0],
        "night_driving": [0.2, 0.3, 0.1, 0.2],
        "rush_hour": [0.5, 0.6, 0.4, 0.5],
        "time_of_day": ["day", "night", "day", "night"],
    }

    trips_df = pd.DataFrame(trips_data)

    driver_features = aggregator.aggregate_trips(trips_df)
    print(driver_features)
