"""Import flagged detections into Label Studio for human review."""

import json
import logging
import os
from pathlib import Path

import yaml

from collector.db import Batch, Classification, Event, get_session

logger = logging.getLogger(__name__)


class LabelStudioImporter:
    """Pushes flagged (uncertain) detections to Label Studio with pre-annotations."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.ls_url = os.environ.get(
            "LABEL_STUDIO_URL",
            self.config["label_studio"]["url"],
        )
        self.api_key = os.environ.get("LABEL_STUDIO_API_KEY", "")
        self.project_name = self.config["label_studio"]["project_name"]
        self.classes = self.config["classes"]["custom"]

        self._client = None

    @property
    def client(self):
        """Lazy-load Label Studio SDK client."""
        if self._client is None:
            from label_studio_sdk import Client

            self._client = Client(url=self.ls_url, api_key=self.api_key)
            # LS 1.14+ disables legacy Token auth in favour of JWT Bearer auth.
            # The SDK still sends "Token <key>" by default, so we override it.
            self._client.session.headers["Authorization"] = f"Bearer {self.api_key}"
            logger.info(f"Connected to Label Studio at {self.ls_url}")
        return self._client

    def get_or_create_project(self):
        """Get existing project or create a new one with object detection config."""
        projects = self.client.get_projects()
        for project in projects:
            if project.title == self.project_name:
                return project

        # Create label config for object detection with our classes
        label_choices = "\n".join(
            f'        <Label value="{cls}" />' for cls in self.classes
        )
        label_config = f"""<View>
  <Image name="image" value="$image" />
  <Header value="Frigate Label: $frigate_label | Score: $frigate_score" />
  <Header value="CLIP: $clip_label ($clip_score) | Google: $google_label ($google_score)" />
  <RectangleLabels name="label" toName="image">
{label_choices}
  </RectangleLabels>
</View>"""

        project = self.client.create_project(
            title=self.project_name,
            label_config=label_config,
        )
        logger.info(f"Created Label Studio project: {self.project_name}")
        return project

    def import_flagged_events(self, batch_id: str | None = None) -> int:
        """Import all flagged-for-review events into Label Studio.

        Args:
            batch_id: Optional batch ID to filter events.

        Returns number of tasks created.
        """
        session = get_session()
        try:
            # Find flagged classifications
            query = (
                session.query(Classification, Event)
                .join(Event, Classification.event_id == Event.id)
                .filter(Classification.decision == "flagged_review")
                .filter(Classification.human_reviewed == False)
            )

            results = query.all()
            if not results:
                logger.info("No flagged events to import")
                return 0

            project = self.get_or_create_project()
            tasks = []

            for classification, event in results:
                # Build the task with pre-annotation
                image_path = event.snapshot_path
                if not image_path or not Path(image_path).exists():
                    logger.warning(f"Snapshot missing for event {event.id}")
                    continue

                # For local files, Label Studio needs the path relative to
                # LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
                # Since we mount ./data to /data, strip the data/ prefix
                relative_path = str(Path(image_path))
                if relative_path.startswith("data/"):
                    relative_path = relative_path[5:]  # Remove "data/" prefix

                # Pre-annotation with suggested bounding box and label
                predictions = []
                if event.box_x is not None:
                    predictions.append(
                        {
                            "model_version": "clip_classifier",
                            "result": [
                                {
                                    "from_name": "label",
                                    "to_name": "image",
                                    "type": "rectanglelabels",
                                    "value": {
                                        "x": event.box_x * 100,
                                        "y": event.box_y * 100,
                                        "width": event.box_w * 100,
                                        "height": event.box_h * 100,
                                        "rectanglelabels": [
                                            classification.final_label
                                        ],
                                    },
                                }
                            ],
                            "score": classification.clip_score or 0.0,
                        }
                    )

                task = {
                    "data": {
                        "image": f"/data/local-files/?d={relative_path}",
                        "frigate_label": event.frigate_label,
                        "frigate_score": f"{event.frigate_score:.2f}",
                        "clip_label": classification.clip_label or "N/A",
                        "clip_score": f"{classification.clip_score:.2f}"
                        if classification.clip_score
                        else "N/A",
                        "google_label": classification.google_label or "N/A",
                        "google_score": f"{classification.google_score:.2f}"
                        if classification.google_score
                        else "N/A",
                        "event_id": event.id,
                        "camera": event.camera,
                    },
                    "predictions": predictions,
                }
                tasks.append(task)

            if tasks:
                project.import_tasks(tasks)
                logger.info(f"Imported {len(tasks)} tasks into Label Studio")

                # Update batch status if applicable
                if batch_id:
                    batch = session.query(Batch).filter_by(id=batch_id).first()
                    if batch:
                        batch.status = "reviewing"
                        batch.label_studio_project_id = project.id
                        session.commit()

            return len(tasks)
        finally:
            session.close()
