"""YOLO11m training with quality gates."""

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from collector.db import TrainingRun, get_session, init_db

logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Fine-tunes YOLO11m on approved camera detection data."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        t_cfg = self.config["training"]
        self.model_base = t_cfg["model_base"]
        self.imgsz = t_cfg["imgsz"]
        self.epochs = t_cfg["epochs"]
        self.batch_size = t_cfg["batch_size"]
        self.patience = t_cfg["patience"]
        self.device = t_cfg["device"]
        self.amp = t_cfg["amp"]
        self.quality_gates = t_cfg["quality_gates"]
        self.classes = self.config["classes"]["custom"]

        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def train(self, dataset_path: str | Path) -> dict:
        """Run YOLO11m fine-tuning.

        Args:
            dataset_path: Path to directory containing dataset.yaml

        Returns:
            Dict with training results and whether quality gates passed.
        """
        from ultralytics import YOLO

        dataset_yaml = Path(dataset_path) / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml}")

        init_db()
        session = get_session()

        # Determine version number
        existing = (
            session.query(TrainingRun)
            .order_by(TrainingRun.id.desc())
            .first()
        )
        version_num = (existing.id + 1) if existing else 1
        version = f"v{version_num:03d}_{datetime.utcnow().strftime('%Y%m%d')}"

        logger.info(f"Starting training run {version}")
        logger.info(f"  Base model: {self.model_base}")
        logger.info(f"  Dataset: {dataset_yaml}")
        logger.info(f"  Config: imgsz={self.imgsz}, batch={self.batch_size}, "
                     f"epochs={self.epochs}, patience={self.patience}")

        # Create training run record
        run = TrainingRun(
            model_base=self.model_base,
            batch_size=self.batch_size,
            version=version,
        )
        session.add(run)
        session.commit()

        try:
            # Load pretrained model
            model = YOLO(self.model_base)

            # Train
            results = model.train(
                data=str(dataset_yaml),
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                amp=self.amp,
                patience=self.patience,
                save_period=10,
                project="runs/train",
                name=f"frigate_{version}",
                exist_ok=True,
            )

            # Extract metrics
            metrics = self._extract_metrics(results)
            passed = self._check_quality_gates(metrics)

            # Save model to versioned directory
            output_dir = self.models_dir / version
            output_dir.mkdir(parents=True, exist_ok=True)

            best_pt = Path(f"runs/train/frigate_{version}/weights/best.pt")
            if best_pt.exists():
                import shutil
                shutil.copy2(best_pt, output_dir / "best.pt")

            # Save metrics
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Update database
            run.completed_at = datetime.utcnow()
            run.epochs_completed = metrics.get("epochs_completed", self.epochs)
            run.map50 = metrics.get("map50")
            run.map50_95 = metrics.get("map50_95")
            run.per_class_map50 = json.dumps(metrics.get("per_class_map50", {}))
            run.model_path = str(output_dir / "best.pt")
            run.passed_quality_gates = passed
            session.commit()

            result = {
                "version": version,
                "model_path": str(output_dir / "best.pt"),
                "metrics": metrics,
                "passed_quality_gates": passed,
            }

            if passed:
                logger.info(f"Training complete. Quality gates PASSED. Model: {output_dir}")
            else:
                logger.warning(
                    f"Training complete but quality gates FAILED. "
                    f"Model saved to {output_dir} but not recommended for deployment."
                )

            return result

        except Exception:
            run.completed_at = datetime.utcnow()
            session.commit()
            logger.exception("Training failed")
            raise
        finally:
            session.close()

    def _extract_metrics(self, results) -> dict:
        """Extract key metrics from YOLO training results."""
        metrics = {}
        try:
            # Results object varies by Ultralytics version
            if hasattr(results, "results_dict"):
                rd = results.results_dict
                metrics["map50"] = rd.get("metrics/mAP50(B)", 0)
                metrics["map50_95"] = rd.get("metrics/mAP50-95(B)", 0)
                metrics["precision"] = rd.get("metrics/precision(B)", 0)
                metrics["recall"] = rd.get("metrics/recall(B)", 0)
            elif hasattr(results, "box"):
                metrics["map50"] = results.box.map50
                metrics["map50_95"] = results.box.map
                metrics["precision"] = results.box.mp
                metrics["recall"] = results.box.mr

            # Per-class metrics
            if hasattr(results, "box") and hasattr(results.box, "maps"):
                per_class = {}
                maps = results.box.maps
                for i, cls_name in enumerate(self.classes):
                    if i < len(maps):
                        per_class[cls_name] = float(maps[i])
                metrics["per_class_map50"] = per_class

        except Exception:
            logger.exception("Failed to extract some metrics")

        return metrics

    def _check_quality_gates(self, metrics: dict) -> bool:
        """Check if training metrics pass quality gates."""
        min_map50 = self.quality_gates["min_map50"]
        min_per_class = self.quality_gates["min_per_class_map50"]

        map50 = metrics.get("map50", 0)
        if map50 < min_map50:
            logger.warning(f"mAP50 {map50:.3f} < minimum {min_map50}")
            return False

        per_class = metrics.get("per_class_map50", {})
        for cls_name, cls_map in per_class.items():
            if cls_map < min_per_class:
                logger.warning(
                    f"Class '{cls_name}' mAP50 {cls_map:.3f} < minimum {min_per_class}"
                )
                return False

        return True
