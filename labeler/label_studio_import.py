"""Import flagged detections into Label Studio for human review."""

import logging
import os
from pathlib import Path

import requests
import yaml

from collector.db import Batch, Classification, Event, get_session

logger = logging.getLogger(__name__)


class LabelStudioImporter:
    """Pushes flagged (uncertain) detections to Label Studio with pre-annotations.

    Auth flow (LS 1.17+):
      LABEL_STUDIO_API_KEY is the non-expiring PAT (JWT refresh token) from
      Account & Settings → Personal Access Token.
      We exchange it for a short-lived access token via POST /api/token/refresh/
      and use that as 'Authorization: Bearer <access>'. On 401 we re-exchange.
    """

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.ls_url = os.environ.get(
            "LABEL_STUDIO_URL",
            self.config["label_studio"]["url"],
        ).rstrip("/")
        # PAT = non-expiring JWT refresh token from the LS UI
        self.pat = os.environ.get("LABEL_STUDIO_API_KEY", "")
        self.project_name = self.config["label_studio"]["project_name"]
        self.classes = self.config["classes"]["custom"]

        self._session = None

    def _refresh_access_token(self):
        """Exchange the PAT for a fresh short-lived access token."""
        resp = requests.post(
            f"{self.ls_url}/api/token/refresh/",
            json={"refresh": self.pat},
            timeout=10,
        )
        resp.raise_for_status()
        access = resp.json()["access"]
        self._session.headers["Authorization"] = f"Bearer {access}"

    @property
    def session(self):
        """Lazy-load HTTP session; exchanges PAT for access token on first use."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
            self._refresh_access_token()
            resp = self._session.get(f"{self.ls_url}/api/current-user/whoami", timeout=10)
            resp.raise_for_status()
            logger.info(f"Connected to Label Studio at {self.ls_url}")
        return self._session

    def _request(self, method: str, path: str, **kwargs):
        """Make an authenticated request, re-exchanging PAT on token expiry."""
        resp = getattr(self.session, method)(f"{self.ls_url}{path}", **kwargs)
        if resp.status_code == 401:
            logger.debug("Access token expired — refreshing")
            self._refresh_access_token()
            resp = getattr(self.session, method)(f"{self.ls_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp

    # ── Low-level API helpers ──────────────────────────────────────────────

    def _get_projects(self) -> list[dict]:
        resp = self._request("get", "/api/projects", timeout=30)
        return resp.json().get("results", resp.json())

    def _create_project(self, title: str, label_config: str) -> dict:
        resp = self._request(
            "post", "/api/projects",
            json={"title": title, "label_config": label_config},
            timeout=30,
        )
        return resp.json()

    def _import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        """Import tasks in chunks of 250 to avoid LS request size limits."""
        total = 0
        chunk_size = 250
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i : i + chunk_size]
            self._request(
                "post", f"/api/projects/{project_id}/import",
                json=chunk,
                timeout=120,
            )
            total += len(chunk)
        return {"imported": total}

    # ── Project management ─────────────────────────────────────────────────

    def get_or_create_project(self) -> dict:
        """Get existing project or create a new one with object detection config."""
        for project in self._get_projects():
            if project.get("title") == self.project_name:
                return project

        # Build label config for object detection with our classes
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

        project = self._create_project(self.project_name, label_config)
        logger.info(f"Created Label Studio project: {self.project_name}")
        return project

    # ── Main import ────────────────────────────────────────────────────────

    def import_flagged_events(self, batch_id: str | None = None) -> int:
        """Import all flagged-for-review events into Label Studio.

        Args:
            batch_id: Optional batch ID to filter events.

        Returns number of tasks created.
        """
        db_session = get_session()
        try:
            query = (
                db_session.query(Classification, Event)
                .join(Event, Classification.event_id == Event.id)
                .filter(Classification.decision == "flagged_review")
                .filter(Classification.human_reviewed == False)
            )

            results = query.all()
            if not results:
                logger.info("No flagged events to import")
                return 0

            project = self.get_or_create_project()
            project_id = project["id"]
            tasks = []

            for classification, event in results:
                image_path = event.snapshot_path
                if not image_path or not Path(image_path).exists():
                    logger.warning(f"Snapshot missing for event {event.id}")
                    continue

                # For local files, Label Studio needs the path relative to
                # LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (/data inside the container).
                # Our snapshots are at data/raw/... so strip the leading "data/".
                relative_path = str(Path(image_path))
                if relative_path.startswith("data/"):
                    relative_path = relative_path[5:]

                # Pre-annotation with suggested bounding box and CLIP label
                predictions = []
                if event.box_x is not None and event.box_w and event.box_w > 0:
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

                tasks.append(
                    {
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
                )

            if tasks:
                self._import_tasks(project_id, tasks)
                logger.info(f"Imported {len(tasks)} tasks into Label Studio")

                if batch_id:
                    batch = db_session.query(Batch).filter_by(id=batch_id).first()
                    if batch:
                        batch.status = "reviewing"
                        batch.label_studio_project_id = project_id
                        db_session.commit()

            return len(tasks)
        finally:
            db_session.close()
