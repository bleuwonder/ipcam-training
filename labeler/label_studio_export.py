"""Export approved annotations from Label Studio."""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from collector.db import Classification, Event, get_session

logger = logging.getLogger(__name__)


class LabelStudioExporter:
    """Pulls completed annotations from Label Studio and updates the database."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.ls_url = os.environ.get(
            "LABEL_STUDIO_URL",
            self.config["label_studio"]["url"],
        )
        self.api_key = os.environ.get("LABEL_STUDIO_API_KEY", "")
        self.project_name = self.config["label_studio"]["project_name"]
        self.approved_dir = Path("data/approved")

        self._client = None

    @property
    def client(self):
        if self._client is None:
            from label_studio_sdk import Client

            self._client = Client(url=self.ls_url, api_key=self.api_key)
        return self._client

    def export_approved(self) -> int:
        """Export completed annotations and update classifications in the DB.

        Returns number of annotations exported.
        """
        projects = self.client.get_projects()
        project = None
        for p in projects:
            if p.title == self.project_name:
                project = p
                break

        if not project:
            logger.warning("Label Studio project not found")
            return 0

        # Get all completed tasks
        tasks = project.get_tasks()
        completed = [t for t in tasks if t.get("is_labeled")]

        if not completed:
            logger.info("No completed annotations to export")
            return 0

        session = get_session()
        exported = 0

        try:
            for task in completed:
                event_id = task["data"].get("event_id")
                if not event_id:
                    continue

                # Get the annotation
                annotations = task.get("annotations", [])
                if not annotations:
                    continue

                annotation = annotations[-1]  # Latest annotation
                results = annotation.get("result", [])

                for result in results:
                    if result.get("type") != "rectanglelabels":
                        continue

                    value = result["value"]
                    human_label = value["rectanglelabels"][0]

                    # Update the classification record
                    classification = (
                        session.query(Classification)
                        .filter_by(event_id=event_id)
                        .first()
                    )
                    if classification:
                        classification.human_label = human_label
                        classification.human_reviewed = True
                        classification.final_label = human_label
                        classification.approved = True
                        classification.approved_at = datetime.utcnow()

                    # Update bbox in event if the reviewer adjusted it
                    event = session.query(Event).filter_by(id=event_id).first()
                    if event:
                        event.box_x = value["x"] / 100.0
                        event.box_y = value["y"] / 100.0
                        event.box_w = value["width"] / 100.0
                        event.box_h = value["height"] / 100.0

                    exported += 1

            session.commit()
            logger.info(f"Exported {exported} annotations from Label Studio")
            return exported
        finally:
            session.close()

    def build_approved_dataset(self) -> Path:
        """Collect all approved data (auto + human) into YOLO format directory.

        Returns path to the approved dataset directory.
        """
        session = get_session()
        try:
            approved = (
                session.query(Classification, Event)
                .join(Event, Classification.event_id == Event.id)
                .filter(Classification.approved == True)
                .all()
            )

            if not approved:
                logger.warning("No approved data found")
                return None

            # Create output directory
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = self.approved_dir / f"dataset_{timestamp}"
            images_dir = output_dir / "images"
            labels_dir = output_dir / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Class index mapping
            classes = self.config["classes"]["custom"]
            class_to_idx = {cls: i for i, cls in enumerate(classes)}

            count = 0
            class_counts = {cls: 0 for cls in classes}

            for classification, event in approved:
                if not event.snapshot_path or not Path(event.snapshot_path).exists():
                    continue

                label = classification.final_label
                if label not in class_to_idx:
                    logger.warning(f"Unknown class {label} for event {event.id}")
                    continue

                # Copy image
                src = Path(event.snapshot_path)
                dst = images_dir / f"{event.id}.png"
                shutil.copy2(src, dst)

                # Write YOLO label file
                # Format: class_idx center_x center_y width height (normalized)
                cx = event.box_x + event.box_w / 2
                cy = event.box_y + event.box_h / 2
                label_file = labels_dir / f"{event.id}.txt"
                label_file.write_text(
                    f"{class_to_idx[label]} {cx:.6f} {cy:.6f} {event.box_w:.6f} {event.box_h:.6f}\n"
                )

                count += 1
                class_counts[label] = class_counts.get(label, 0) + 1

            # Write dataset.yaml
            dataset_yaml = {
                "path": str(output_dir.resolve()),
                "train": "images",
                "val": "images",
                "names": {i: name for i, name in enumerate(classes)},
            }
            with open(output_dir / "dataset.yaml", "w") as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)

            logger.info(
                f"Built approved dataset with {count} images at {output_dir}. "
                f"Class distribution: {class_counts}"
            )
            return output_dir
        finally:
            session.close()
