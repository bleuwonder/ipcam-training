"""Model evaluation — compare against previous models and generate reports."""

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a trained YOLO model and compares against previous versions."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.classes = self.config["classes"]["custom"]
        self.models_dir = Path("models")

    def evaluate(self, model_path: str | Path, dataset_path: str | Path) -> dict:
        """Run validation on a model and return metrics.

        Args:
            model_path: Path to the .pt model file.
            dataset_path: Path to dataset directory with dataset.yaml.
        """
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        dataset_yaml = Path(dataset_path) / "dataset.yaml"

        results = model.val(data=str(dataset_yaml))

        metrics = {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

        # Per-class metrics
        if hasattr(results.box, "maps"):
            per_class = {}
            for i, cls_name in enumerate(self.classes):
                if i < len(results.box.maps):
                    per_class[cls_name] = float(results.box.maps[i])
            metrics["per_class_map50"] = per_class

        return metrics

    def compare_versions(self) -> list[dict]:
        """Load metrics from all model versions for comparison."""
        versions = []
        for version_dir in sorted(self.models_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            metrics_file = version_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                versions.append(
                    {"version": version_dir.name, "metrics": metrics}
                )
        return versions

    def print_report(self, metrics: dict, version: str = "current"):
        """Print a formatted evaluation report."""
        print(f"\n{'=' * 60}")
        print(f"  Model Evaluation Report: {version}")
        print(f"{'=' * 60}")
        print(f"  mAP@50:      {metrics.get('map50', 0):.4f}")
        print(f"  mAP@50-95:   {metrics.get('map50_95', 0):.4f}")
        print(f"  Precision:   {metrics.get('precision', 0):.4f}")
        print(f"  Recall:      {metrics.get('recall', 0):.4f}")

        per_class = metrics.get("per_class_map50", {})
        if per_class:
            print(f"\n  Per-class mAP@50:")
            for cls_name, map50 in per_class.items():
                status = "PASS" if map50 >= 0.50 else "FAIL"
                print(f"    {cls_name:12s}: {map50:.4f}  [{status}]")

        # Quality gate check
        gates = self.config["training"]["quality_gates"]
        overall_pass = metrics.get("map50", 0) >= gates["min_map50"]
        class_pass = all(
            v >= gates["min_per_class_map50"] for v in per_class.values()
        ) if per_class else True

        print(f"\n  Quality Gates:")
        print(f"    Overall mAP50 >= {gates['min_map50']}: "
              f"{'PASS' if overall_pass else 'FAIL'}")
        print(f"    Per-class >= {gates['min_per_class_map50']}: "
              f"{'PASS' if class_pass else 'FAIL'}")
        print(f"{'=' * 60}\n")
