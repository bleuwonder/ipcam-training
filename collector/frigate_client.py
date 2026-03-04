"""Frigate HTTP API client for fetching events and snapshots."""

import os
from dataclasses import dataclass
from datetime import datetime

import requests


@dataclass
class FrigateEvent:
    """Parsed Frigate event."""

    id: str
    camera: str
    label: str
    score: float
    box: tuple[float, float, float, float]  # x, y, w, h (normalized 0-1)
    start_time: datetime
    has_snapshot: bool


class FrigateClient:
    """Client for Frigate's HTTP API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url or os.environ.get("FRIGATE_URL", "http://localhost:5000")
        ).rstrip("/")
        self.session = requests.Session()

    def get_events(
        self,
        after: float | None = None,
        labels: list[str] | None = None,
        cameras: list[str] | None = None,
        has_snapshot: bool = True,
        limit: int = 100,
    ) -> list[FrigateEvent]:
        """Fetch events from Frigate API.

        Args:
            after: Unix timestamp — only return events after this time.
            labels: Filter by object labels (e.g. ["person", "car"]).
            cameras: Filter by camera names.
            has_snapshot: Only return events with snapshots.
            limit: Max number of events to return.
        """
        params = {"limit": limit, "has_snapshot": int(has_snapshot)}
        if after is not None:
            params["after"] = after
        if labels:
            params["labels"] = ",".join(labels)
        if cameras:
            params["cameras"] = ",".join(cameras)

        resp = self.session.get(f"{self.base_url}/api/events", params=params)
        resp.raise_for_status()

        events = []
        for raw in resp.json():
            # Frigate returns box as [y1, x1, y2, x2] normalized
            # Convert to [x, y, w, h]
            raw_box = raw.get("box", [0, 0, 0, 0])
            if len(raw_box) == 4:
                y1, x1, y2, x2 = raw_box
                box = (x1, y1, x2 - x1, y2 - y1)
            else:
                box = (0, 0, 0, 0)

            events.append(
                FrigateEvent(
                    id=raw["id"],
                    camera=raw["camera"],
                    label=raw["label"],
                    score=raw.get("top_score", raw.get("score", 0.0)),
                    box=box,
                    start_time=datetime.fromtimestamp(raw["start_time"]),
                    has_snapshot=raw.get("has_snapshot", False),
                )
            )
        return events

    def download_snapshot_clean(self, event_id: str) -> bytes:
        """Download the clean (unannotated) snapshot for an event."""
        resp = self.session.get(
            f"{self.base_url}/api/events/{event_id}/snapshot-clean.png"
        )
        resp.raise_for_status()
        return resp.content

    def download_snapshot(self, event_id: str) -> bytes:
        """Download the annotated snapshot for an event."""
        resp = self.session.get(
            f"{self.base_url}/api/events/{event_id}/snapshot.jpg"
        )
        resp.raise_for_status()
        return resp.content

    def get_camera_snapshot(self, camera_name: str) -> bytes:
        """Get a current snapshot from a camera (for package discovery)."""
        resp = self.session.get(
            f"{self.base_url}/api/{camera_name}/latest.jpg",
            params={"quality": 95},
        )
        resp.raise_for_status()
        return resp.content

    def health_check(self) -> bool:
        """Check if Frigate is reachable."""
        try:
            resp = self.session.get(f"{self.base_url}/api/version", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False
