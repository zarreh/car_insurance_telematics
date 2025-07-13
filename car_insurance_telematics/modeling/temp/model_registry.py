"""
Model Registry
Model versioning and management system for telematics ML models
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model versioning and management system"""

    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize the model registry

        Args:
            registry_dir: Directory to store model registry
        """
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "registry.json")

        # Create registry directory if it doesn't exist
        os.makedirs(registry_dir, exist_ok=True)

        # Load or create registry
        self.registry = self._load_registry()

        logger.info(f"Model Registry initialized at {registry_dir}")

    def _load_registry(self) -> Dict[str, Any]:
        """Load existing registry or create new one"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "metadata": {"created_at": datetime.now().isoformat(), "last_updated": datetime.now().isoformat()},
            }

    def _save_registry(self):
        """Save registry to disk"""
        self.registry["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def save_model(self, model: Any, model_type: str, model_info: Dict[str, Any]) -> str:
        """
        Save a model to the registry

        Args:
            model: The model object to save
            model_type: Type of model ('claim_probability' or 'claim_severity')
            model_info: Dictionary containing model information

        Returns:
            Path to the saved model
        """
        # Generate version number
        if model_type not in self.registry["models"]:
            self.registry["models"][model_type] = {}

        existing_versions = list(self.registry["models"][model_type].keys())
        if existing_versions:
            latest_version = max([int(v.replace("v", "")) for v in existing_versions])
            version = f"v{latest_version + 1}"
        else:
            version = "v1"

        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_type, version)
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        if hasattr(model, "save"):
            model.save(model_path)
        else:
            joblib.dump(model, model_path)

        # Save model info
        info_path = os.path.join(model_dir, "model_info.json")
        model_info["version"] = version
        model_info["model_path"] = model_path
        model_info["created_at"] = datetime.now().isoformat()

        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        # Update registry
        self.registry["models"][model_type][version] = {
            "path": model_path,
            "info_path": info_path,
            "created_at": model_info["created_at"],
            "metrics": model_info.get("metrics", {}),
            "algorithm": model_info.get("algorithm", "unknown"),
        }

        # Mark as latest
        self.registry["models"][model_type]["latest"] = version

        self._save_registry()

        logger.info(f"Model saved: {model_type} {version} at {model_path}")

        return model_path

    def load_model(self, model_type: str, version: Optional[str] = None) -> Any:
        """
        Load a model from the registry

        Args:
            model_type: Type of model to load
            version: Specific version to load (default: latest)

        Returns:
            Loaded model object
        """
        if model_type not in self.registry["models"]:
            raise ValueError(f"No models found for type: {model_type}")

        if version is None:
            version = self.registry["models"][model_type].get("latest")
            if version is None:
                raise ValueError(f"No latest version found for {model_type}")

        if version not in self.registry["models"][model_type]:
            raise ValueError(f"Version {version} not found for {model_type}")

        model_path = self.registry["models"][model_type][version]["path"]

        # Try to load using model's own load method first
        try:
            # Import the appropriate model class
            if model_type == "claim_probability":
                from claim_probability_model import ClaimProbabilityModel

                model = ClaimProbabilityModel()
                model.load(model_path)
            elif model_type == "claim_severity":
                from claim_severity_model import ClaimSeverityModel

                model = ClaimSeverityModel()
                model.load(model_path)
            else:
                model = joblib.load(model_path)
        except:
            # Fallback to joblib
            model = joblib.load(model_path)

        logger.info(f"Model loaded: {model_type} {version} from {model_path}")

        return model

    def get_model_info(self, model_type: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model

        Args:
            model_type: Type of model
            version: Specific version (default: latest)

        Returns:
            Model information dictionary
        """
        if model_type not in self.registry["models"]:
            raise ValueError(f"No models found for type: {model_type}")

        if version is None:
            version = self.registry["models"][model_type].get("latest")

        if version not in self.registry["models"][model_type]:
            raise ValueError(f"Version {version} not found for {model_type}")

        info_path = self.registry["models"][model_type][version]["info_path"]

        with open(info_path, "r") as f:
            return json.load(f)

    def list_models(self, model_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all models in the registry

        Args:
            model_type: Filter by model type (optional)

        Returns:
            Dictionary of model types and their versions
        """
        if model_type:
            if model_type in self.registry["models"]:
                versions = [v for v in self.registry["models"][model_type].keys() if v != "latest"]
                return {model_type: versions}
            else:
                return {model_type: []}
        else:
            result = {}
            for mt in self.registry["models"]:
                versions = [v for v in self.registry["models"][mt].keys() if v != "latest"]
                result[mt] = versions
            return result

    def compare_versions(self, model_type: str) -> pd.DataFrame:
        """
        Compare different versions of a model type

        Args:
            model_type: Type of model to compare

        Returns:
            DataFrame with version comparison
        """
        if model_type not in self.registry["models"]:
            raise ValueError(f"No models found for type: {model_type}")

        comparison_data = []

        for version, info in self.registry["models"][model_type].items():
            if version == "latest":
                continue

            row = {"version": version, "created_at": info["created_at"], "algorithm": info.get("algorithm", "unknown")}

            # Add metrics
            metrics = info.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    row[metric_name] = metric_value

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("version")

        return df

    def promote_to_production(self, model_type: str, version: str):
        """
        Promote a specific model version to production

        Args:
            model_type: Type of model
            version: Version to promote
        """
        if model_type not in self.registry["models"]:
            raise ValueError(f"No models found for type: {model_type}")

        if version not in self.registry["models"][model_type]:
            raise ValueError(f"Version {version} not found for {model_type}")

        # Create production directory
        prod_dir = os.path.join(self.registry_dir, "production")
        os.makedirs(prod_dir, exist_ok=True)

        # Copy model to production
        source_path = self.registry["models"][model_type][version]["path"]
        dest_path = os.path.join(prod_dir, f"{model_type}_model.pkl")
        shutil.copy2(source_path, dest_path)

        # Copy model info
        source_info = self.registry["models"][model_type][version]["info_path"]
        dest_info = os.path.join(prod_dir, f"{model_type}_model_info.json")
        shutil.copy2(source_info, dest_info)

        # Update production metadata
        prod_metadata = {
            "model_type": model_type,
            "version": version,
            "promoted_at": datetime.now().isoformat(),
            "source_path": source_path,
            "production_path": dest_path,
        }

        prod_metadata_path = os.path.join(prod_dir, f"{model_type}_production.json")
        with open(prod_metadata_path, "w") as f:
            json.dump(prod_metadata, f, indent=2)

        logger.info(f"Model promoted to production: {model_type} {version}")

    def get_production_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about production models

        Returns:
            Dictionary of production model information
        """
        prod_dir = os.path.join(self.registry_dir, "production")
        if not os.path.exists(prod_dir):
            return {}

        production_models = {}

        for file in os.listdir(prod_dir):
            if file.endswith("_production.json"):
                model_type = file.replace("_production.json", "")
                with open(os.path.join(prod_dir, file), "r") as f:
                    production_models[model_type] = json.load(f)

        return production_models

    def delete_version(self, model_type: str, version: str):
        """
        Delete a specific model version

        Args:
            model_type: Type of model
            version: Version to delete
        """
        if model_type not in self.registry["models"]:
            raise ValueError(f"No models found for type: {model_type}")

        if version not in self.registry["models"][model_type]:
            raise ValueError(f"Version {version} not found for {model_type}")

        if version == self.registry["models"][model_type].get("latest"):
            raise ValueError("Cannot delete the latest version. Promote another version first.")

        # Delete model directory
        model_dir = os.path.dirname(self.registry["models"][model_type][version]["path"])
        shutil.rmtree(model_dir)

        # Remove from registry
        del self.registry["models"][model_type][version]
        self._save_registry()

        logger.info(f"Model deleted: {model_type} {version}")

    def export_registry_report(self) -> str:
        """
        Export a comprehensive registry report

        Returns:
            Path to the report file
        """
        report_path = os.path.join(self.registry_dir, "registry_report.txt")

        with open(report_path, "w") as f:
            f.write("Model Registry Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write(f"Registry location: {self.registry_dir}\n\n")

            # Summary
            f.write("Summary:\n")
            f.write("-" * 30 + "\n")
            for model_type in self.registry["models"]:
                versions = [v for v in self.registry["models"][model_type].keys() if v != "latest"]
                latest = self.registry["models"][model_type].get("latest", "None")
                f.write(f"{model_type}: {len(versions)} versions, latest: {latest}\n")

            # Production models
            f.write("\nProduction Models:\n")
            f.write("-" * 30 + "\n")
            prod_models = self.get_production_models()
            if prod_models:
                for model_type, info in prod_models.items():
                    f.write(f"{model_type}: {info['version']} ")
                    f.write(f"(promoted at {info['promoted_at']})\n")
            else:
                f.write("No models in production\n")

            # Detailed model information
            f.write("\nDetailed Model Information:\n")
            f.write("=" * 60 + "\n")

            for model_type in self.registry["models"]:
                f.write(f"\n{model_type.upper()}:\n")
                f.write("-" * 30 + "\n")

                for version, info in sorted(self.registry["models"][model_type].items()):
                    if version == "latest":
                        continue

                    f.write(f"\n{version}:\n")
                    f.write(f"  Created: {info['created_at']}\n")
                    f.write(f"  Algorithm: {info.get('algorithm', 'unknown')}\n")

                    metrics = info.get("metrics", {})
                    if metrics:
                        f.write("  Metrics:\n")
                        for metric, value in sorted(metrics.items()):
                            if isinstance(value, float):
                                f.write(f"    {metric}: {value:.4f}\n")
                            else:
                                f.write(f"    {metric}: {value}\n")

        logger.info(f"Registry report exported to {report_path}")
        return report_path


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()

    # List all models
    print("All models:", registry.list_models())

    # Get production models
    print("Production models:", registry.get_production_models())
