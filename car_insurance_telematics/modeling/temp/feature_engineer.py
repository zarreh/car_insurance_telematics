"""
Feature Engineer
Feature engineering module for telematics data
"""

import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for telematics data"""

    def __init__(self):
        """Initialize the feature engineer"""
        self.feature_names = []
        self.feature_stats = {}
        logger.info("FeatureEngineer initialized")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw telematics data

        Args:
            df: DataFrame with raw telematics data

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Creating features for {len(df)} records")

        # Create a copy to avoid modifying original data
        features_df = pd.DataFrame()

        # Basic trip features
        basic_features = self._create_basic_features(df)
        features_df = pd.concat([features_df, basic_features], axis=1)

        # Speed-related features
        speed_features = self._create_speed_features(df)
        features_df = pd.concat([features_df, speed_features], axis=1)

        # Driving behavior features
        behavior_features = self._create_behavior_features(df)
        features_df = pd.concat([features_df, behavior_features], axis=1)

        # Time-based features
        time_features = self._create_time_features(df)
        features_df = pd.concat([features_df, time_features], axis=1)

        # Risk indicator features
        risk_features = self._create_risk_features(df)
        features_df = pd.concat([features_df, risk_features], axis=1)

        # Interaction features
        interaction_features = self._create_interaction_features(features_df)
        features_df = pd.concat([features_df, interaction_features], axis=1)

        # Store feature names
        self.feature_names = list(features_df.columns)

        # Calculate feature statistics
        self._calculate_feature_stats(features_df)

        logger.info(f"Created {len(self.feature_names)} features")

        return features_df

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic trip features"""
        features = pd.DataFrame()

        # Duration and distance
        features["trip_duration_minutes"] = df.get("trip_duration_minutes", 0)
        features["trip_distance_km"] = df.get("trip_distance_km", 0)

        # Average metrics
        features["avg_trip_speed"] = np.where(
            features["trip_duration_minutes"] > 0,
            features["trip_distance_km"] / (features["trip_duration_minutes"] / 60),
            0,
        )

        # Log transformations for skewed features
        features["log_trip_duration"] = np.log1p(features["trip_duration_minutes"])
        features["log_trip_distance"] = np.log1p(features["trip_distance_km"])

        # Trip efficiency
        features["trip_efficiency"] = np.where(
            features["trip_distance_km"] > 0, features["trip_duration_minutes"] / features["trip_distance_km"], 0
        )

        return features

    def _create_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create speed-related features"""
        features = pd.DataFrame()

        # Basic speed metrics
        features["max_speed_kmh"] = df.get("max_speed_kmh", 0)
        features["avg_speed_kmh"] = df.get("avg_speed_kmh", 0)

        # Speed variability
        features["speed_variance"] = np.where(
            features["avg_speed_kmh"] > 0,
            (features["max_speed_kmh"] - features["avg_speed_kmh"]) / features["avg_speed_kmh"],
            0,
        )

        # Speed categories
        features["high_speed_indicator"] = (features["max_speed_kmh"] > 120).astype(int)
        features["very_high_speed_indicator"] = (features["max_speed_kmh"] > 140).astype(int)

        # Speed consistency
        features["speed_consistency"] = np.where(
            features["max_speed_kmh"] > 0, features["avg_speed_kmh"] / features["max_speed_kmh"], 0
        )

        # Speeding severity
        speed_limit_estimate = 100  # Assumed general speed limit
        features["speeding_severity"] = np.maximum(
            0, (features["max_speed_kmh"] - speed_limit_estimate) / speed_limit_estimate
        )

        return features

    def _create_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create driving behavior features"""
        features = pd.DataFrame()

        # Aggressive driving events
        features["hard_braking_count"] = df.get("hard_braking_count", 0)
        features["hard_acceleration_count"] = df.get("hard_acceleration_count", 0)
        features["sharp_turn_count"] = df.get("sharp_turn_count", 0)

        # Total aggressive events
        features["total_aggressive_events"] = (
            features["hard_braking_count"] + features["hard_acceleration_count"] + features["sharp_turn_count"]
        )

        # Aggressive driving rates (per km)
        distance = df.get("trip_distance_km", 1).clip(lower=0.1)  # Avoid division by zero
        features["hard_braking_rate"] = features["hard_braking_count"] / distance
        features["hard_acceleration_rate"] = features["hard_acceleration_count"] / distance
        features["aggressive_events_rate"] = features["total_aggressive_events"] / distance

        # Aggressive driving rates (per hour)
        duration_hours = (df.get("trip_duration_minutes", 1) / 60).clip(lower=0.01)
        features["hard_braking_per_hour"] = features["hard_braking_count"] / duration_hours
        features["hard_acceleration_per_hour"] = features["hard_acceleration_count"] / duration_hours

        # Binary indicators
        features["has_hard_braking"] = (features["hard_braking_count"] > 0).astype(int)
        features["has_hard_acceleration"] = (features["hard_acceleration_count"] > 0).astype(int)
        features["has_aggressive_driving"] = (features["total_aggressive_events"] > 0).astype(int)

        # Driving smoothness score (inverse of aggressiveness)
        features["driving_smoothness"] = 1 / (1 + features["total_aggressive_events"])

        return features

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame()

        # Time of day features
        features["night_driving_minutes"] = df.get("night_driving_minutes", 0)
        features["rush_hour_minutes"] = df.get("rush_hour_minutes", 0)

        # Proportions
        total_duration = df.get("trip_duration_minutes", 1).clip(lower=1)
        features["night_driving_ratio"] = features["night_driving_minutes"] / total_duration
        features["rush_hour_ratio"] = features["rush_hour_minutes"] / total_duration

        # Binary indicators
        features["has_night_driving"] = (features["night_driving_minutes"] > 0).astype(int)
        features["majority_night_driving"] = (features["night_driving_ratio"] > 0.5).astype(int)
        features["has_rush_hour_driving"] = (features["rush_hour_minutes"] > 0).astype(int)

        # Weekend indicator
        features["weekend_driving"] = df.get("weekend_driving", 0)

        # Time risk score
        features["time_risk_score"] = (
            features["night_driving_ratio"] * 0.3
            + features["rush_hour_ratio"] * 0.2
            + features["weekend_driving"] * 0.1
        )

        return features

    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicator features"""
        features = pd.DataFrame()

        # Phone usage
        features["phone_use_minutes"] = df.get("phone_use_minutes", 0)
        features["phone_use_ratio"] = features["phone_use_minutes"] / df.get("trip_duration_minutes", 1).clip(lower=1)
        features["has_phone_use"] = (features["phone_use_minutes"] > 0).astype(int)
        features["excessive_phone_use"] = (features["phone_use_ratio"] > 0.1).astype(int)

        # Fatigue indicators
        features["continuous_driving_hours"] = df.get("continuous_driving_hours", 0)
        features["fatigue_risk"] = (features["continuous_driving_hours"] > 2).astype(int)
        features["high_fatigue_risk"] = (features["continuous_driving_hours"] > 4).astype(int)

        # Weather conditions (if available)
        features["adverse_weather"] = df.get("adverse_weather", 0)
        features["rain_driving"] = df.get("rain_driving", 0)
        features["snow_driving"] = df.get("snow_driving", 0)
        features["fog_driving"] = df.get("fog_driving", 0)

        # Combined weather risk
        features["weather_risk_score"] = (
            features["adverse_weather"] * 0.2
            + features["rain_driving"] * 0.3
            + features["snow_driving"] * 0.4
            + features["fog_driving"] * 0.3
        )

        # Traffic density
        features["high_traffic_ratio"] = df.get("high_traffic_ratio", 0)
        features["traffic_risk"] = (features["high_traffic_ratio"] > 0.5).astype(int)

        # Overall risk indicators
        features["total_risk_factors"] = (
            features["has_phone_use"]
            + features["fatigue_risk"]
            + features["adverse_weather"]
            + features["traffic_risk"]
        )

        return features

    def _create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between existing features"""
        features = pd.DataFrame()

        # Speed and aggressive driving interaction
        features["speed_aggression_interaction"] = (
            features_df.get("max_speed_kmh", 0) * features_df.get("total_aggressive_events", 0) / 100
        )

        # Night driving and speed interaction
        features["night_speed_risk"] = (
            features_df.get("night_driving_ratio", 0) * features_df.get("max_speed_kmh", 0) / 100
        )

        # Distance and aggressive events interaction
        features["long_trip_aggression"] = features_df.get("log_trip_distance", 0) * features_df.get(
            "aggressive_events_rate", 0
        )

        # Phone use and speed interaction
        features["distracted_speeding"] = features_df.get("phone_use_ratio", 0) * features_df.get(
            "speeding_severity", 0
        )

        # Fatigue and time of day interaction
        features["fatigue_night_risk"] = features_df.get("continuous_driving_hours", 0) * features_df.get(
            "night_driving_ratio", 0
        )

        # Weather and aggressive driving interaction
        features["weather_aggression_risk"] = features_df.get("weather_risk_score", 0) * features_df.get(
            "total_aggressive_events", 0
        )

        # Composite risk score
        features["composite_risk_score"] = (
            features_df.get("speed_variance", 0) * 0.15
            + features_df.get("aggressive_events_rate", 0) * 0.25
            + features_df.get("time_risk_score", 0) * 0.20
            + features_df.get("phone_use_ratio", 0) * 0.20
            + features_df.get("weather_risk_score", 0) * 0.20
        ).clip(upper=1.0)

        return features

    def _calculate_feature_stats(self, features_df: pd.DataFrame):
        """Calculate and store feature statistics"""
        self.feature_stats = {
            "mean": features_df.mean().to_dict(),
            "std": features_df.std().to_dict(),
            "min": features_df.min().to_dict(),
            "max": features_df.max().to_dict(),
            "missing_ratio": features_df.isnull().mean().to_dict(),
        }

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names

    def get_feature_importance_hints(self) -> Dict[str, float]:
        """
        Get hints about expected feature importance based on domain knowledge

        Returns:
            Dictionary of feature names and importance hints (0-1)
        """
        importance_hints = {
            # High importance features
            "total_aggressive_events": 0.9,
            "aggressive_events_rate": 0.9,
            "max_speed_kmh": 0.8,
            "speeding_severity": 0.8,
            "composite_risk_score": 0.85,
            "phone_use_ratio": 0.75,
            # Medium importance features
            "night_driving_ratio": 0.6,
            "hard_braking_rate": 0.65,
            "hard_acceleration_rate": 0.65,
            "speed_variance": 0.6,
            "weather_risk_score": 0.55,
            "fatigue_risk": 0.5,
            # Lower importance features
            "trip_distance_km": 0.4,
            "trip_duration_minutes": 0.4,
            "weekend_driving": 0.3,
            "rush_hour_ratio": 0.35,
            # Interaction features
            "speed_aggression_interaction": 0.7,
            "night_speed_risk": 0.65,
            "distracted_speeding": 0.7,
            "weather_aggression_risk": 0.6,
        }

        # Fill in remaining features with default importance
        for feature in self.feature_names:
            if feature not in importance_hints:
                importance_hints[feature] = 0.3

        return importance_hints

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions of features"""
        descriptions = {
            # Basic features
            "trip_duration_minutes": "Total duration of the trip in minutes",
            "trip_distance_km": "Total distance traveled in kilometers",
            "avg_trip_speed": "Average speed for the entire trip (km/h)",
            "log_trip_duration": "Log-transformed trip duration",
            "log_trip_distance": "Log-transformed trip distance",
            "trip_efficiency": "Minutes per kilometer (inverse of average speed)",
            # Speed features
            "max_speed_kmh": "Maximum speed reached during the trip (km/h)",
            "avg_speed_kmh": "Average speed during the trip (km/h)",
            "speed_variance": "Relative difference between max and average speed",
            "high_speed_indicator": "Binary flag for speeds over 120 km/h",
            "very_high_speed_indicator": "Binary flag for speeds over 140 km/h",
            "speed_consistency": "Ratio of average to maximum speed",
            "speeding_severity": "Degree of speeding above assumed limit",
            # Behavior features
            "hard_braking_count": "Number of hard braking events",
            "hard_acceleration_count": "Number of hard acceleration events",
            "sharp_turn_count": "Number of sharp turn events",
            "total_aggressive_events": "Sum of all aggressive driving events",
            "hard_braking_rate": "Hard braking events per kilometer",
            "hard_acceleration_rate": "Hard acceleration events per kilometer",
            "aggressive_events_rate": "Total aggressive events per kilometer",
            "driving_smoothness": "Inverse measure of driving aggressiveness",
            # Time features
            "night_driving_minutes": "Minutes driven during night hours",
            "night_driving_ratio": "Proportion of trip during night hours",
            "rush_hour_minutes": "Minutes driven during rush hours",
            "rush_hour_ratio": "Proportion of trip during rush hours",
            "weekend_driving": "Binary flag for weekend trips",
            "time_risk_score": "Combined risk score based on time factors",
            # Risk features
            "phone_use_minutes": "Minutes of phone usage while driving",
            "phone_use_ratio": "Proportion of trip with phone usage",
            "continuous_driving_hours": "Hours of continuous driving without break",
            "fatigue_risk": "Binary flag for fatigue risk (>2 hours continuous)",
            "weather_risk_score": "Combined score for adverse weather conditions",
            "composite_risk_score": "Overall risk score combining multiple factors",
        }

        return descriptions

    def validate_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature quality and identify potential issues

        Args:
            features_df: DataFrame with features to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {"valid": True, "issues": [], "warnings": [], "statistics": {}}

        # Check for missing values
        missing_ratios = features_df.isnull().mean()
        high_missing = missing_ratios[missing_ratios > 0.1]
        if len(high_missing) > 0:
            validation_results["warnings"].append(f"High missing values in features: {list(high_missing.index)}")

        # Check for constant features
        constant_features = []
        for col in features_df.columns:
            if features_df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            validation_results["warnings"].append(f"Constant features detected: {constant_features}")

        # Check for infinite values
        inf_features = []
        for col in features_df.select_dtypes(include=[np.number]).columns:
            if np.isinf(features_df[col]).any():
                inf_features.append(col)

        if inf_features:
            validation_results["issues"].append(f"Infinite values in features: {inf_features}")
            validation_results["valid"] = False

        # Check feature ranges
        for col in features_df.select_dtypes(include=[np.number]).columns:
            col_min = features_df[col].min()
            col_max = features_df[col].max()

            # Check for unexpected negative values in count features
            if "count" in col or "minutes" in col or "hours" in col:
                if col_min < 0:
                    validation_results["issues"].append(f"Negative values in {col}: min={col_min}")
                    validation_results["valid"] = False

            # Check for unrealistic values
            if "ratio" in col or "indicator" in col:
                if col_min < 0 or col_max > 1:
                    validation_results["warnings"].append(f"Values outside [0,1] in {col}: [{col_min}, {col_max}]")

        # Add statistics
        validation_results["statistics"] = {
            "n_features": len(features_df.columns),
            "n_samples": len(features_df),
            "missing_features": len(high_missing),
            "constant_features": len(constant_features),
            "numeric_features": len(features_df.select_dtypes(include=[np.number]).columns),
        }

        return validation_results


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()

    # Sample data
    sample_data = pd.DataFrame(
        {
            "trip_duration_minutes": [45, 30, 60],
            "trip_distance_km": [30, 20, 50],
            "max_speed_kmh": [120, 90, 110],
            "avg_speed_kmh": [80, 60, 85],
            "hard_braking_count": [2, 0, 1],
            "hard_acceleration_count": [1, 0, 2],
            "night_driving_minutes": [10, 0, 30],
            "phone_use_minutes": [5, 0, 2],
        }
    )

    features = engineer.create_features(sample_data)
    print(f"Created {len(features.columns)} features")
    print("Feature names:", engineer.get_feature_names()[:10])
    # print("Feature statistics:", engineer.feature_stats)
    print("Feature importance hints:", engineer.get_feature_importance_hints())
    print("Feature descriptions:", engineer.get_feature_descriptions())
    validation = engineer.validate_features(features)
    print("Validation results:", validation)
