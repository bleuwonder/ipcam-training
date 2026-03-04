"""Dataset preparation — merges approved data and creates train/val splits."""

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from collector.db import Classification, Event, get_session

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds a YOLO-format training dataset from all approved annotations."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.classes = self.config["classes"]["custom"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.training_dir = Path("data/training_set")
        self.min_images = self.config["training"]["min_training_images"]
        self.min_per_class = self.config["training"]["min_per_class"]

    def build(self, val_split: float = 0.2, seed: int = 42) -> Path | None:
        """Build consolidated training dataset from all approved data.

        Args:
            val_split: Fraction of data for validation (default 0.2).
            seed: Random seed for reproducible splits.

        Returns:
            Path to dataset directory with dataset.yaml, or None if insufficient data.
        """
        session = get_session()
        try:
            # Gather all approved classifications with events
            approved = (
                session.query(Classification, Event)
                .join(Event, Classification.event_id == Event.id)
                .filter(Classification.approved == True)
                .all()
            )

            if not approved:
                logger.error("No approved data found")
                return None

            # Group by class for stratified splitting
            by_class = defaultdict(list)
            for classification, event in approved:
                label = classification.final_label
                if label in self.class_to_idx and event.snapshot_path:
                    if Path(event.snapshot_path).exists():
                        by_class[label].append((classification, event))

            # Check minimums
            total = sum(len(v) for v in by_class.values())
            logger.info(f"Total approved samples: {total}")
            for cls in self.classes:
                count = len(by_class.get(cls, []))
                logger.info(f"  {cls}: {count} samples")
                if count < self.min_per_class:
                    logger.warning(
                        f"Class '{cls}' has {count} samples "
                        f"(minimum: {self.min_per_class}). "
                        f"Continue collecting data."
                    )

            if total < self.min_images:
                logger.error(
                    f"Only {total} approved images "
                    f"(minimum: {self.min_images}). Need more data."
                )
                return None

            # Clear and recreate training directory
            if self.training_dir.exists():
                shutil.rmtree(self.training_dir)

            train_images = self.training_dir / "images" / "train"
            train_labels = self.training_dir / "labels" / "train"
            val_images = self.training_dir / "images" / "val"
            val_labels = self.training_dir / "labels" / "val"

            for d in [train_images, train_labels, val_images, val_labels]:
                d.mkdir(parents=True)

            # Stratified split
            random.seed(seed)
            train_count = 0
            val_count = 0

            for cls, items in by_class.items():
                random.shuffle(items)
                split_idx = max(1, int(len(items) * (1 - val_split)))
                train_items = items[:split_idx]
                val_items = items[split_idx:]

                for classification, event in train_items:
                    self._write_sample(
                        event, classification, train_images, train_labels
                    )
                    train_count += 1

                for classification, event in val_items:
                    self._write_sample(
                        event, classification, val_images, val_labels
                    )
                    val_count += 1

            # Write dataset.yaml
            dataset_config = {
                "path": str(self.training_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {i: name for i, name in enumerate(self.classes)},
            }
            dataset_yaml_path = self.training_dir / "dataset.yaml"
            with open(dataset_yaml_path, "w") as f:
                yaml.dump(dataset_config, f, default_flow_style=False)

            logger.info(
                f"Dataset built: {train_count} train, {val_count} val "
                f"at {self.training_dir}"
            )
            return self.training_dir

        finally:
            session.close()

    def _write_sample(self, event, classification, images_dir, labels_dir):
        """Copy image and write YOLO label file for a single sample."""
        src = Path(event.snapshot_path)
        dst = images_dir / f"{event.id}.png"
        shutil.copy2(src, dst)

        label = classification.final_label
        class_idx = self.class_to_idx[label]

        # YOLO format: class_idx center_x center_y width height (normalized)
        cx = event.box_x + event.box_w / 2
        cy = event.box_y + event.box_h / 2

        label_file = labels_dir / f"{event.id}.txt"
        label_file.write_text(
            f"{class_idx} {cx:.6f} {cy:.6f} {event.box_w:.6f} {event.box_h:.6f}\n"
        )

    def get_stats(self) -> dict:
        """Get current dataset statistics."""
        session = get_session()
        try:
            approved = (
                session.query(Classification)
                .filter(Classification.approved == True)
                .all()
            )

            stats = {
                "total_approved": len(approved),
                "by_class": defaultdict(int),
                "by_decision": defaultdict(int),
            }
            for c in approved:
                stats["by_class"][c.final_label] += 1
                stats["by_decision"][c.decision] += 1

            return dict(stats)
        finally:
            session.close()
