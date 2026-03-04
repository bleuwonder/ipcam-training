"""Snapshot collector daemon — polls Frigate for new events and saves clean snapshots."""

import io
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

from collector.db import Batch, Event, get_session, init_db
from collector.frigate_client import FrigateClient

logger = logging.getLogger(__name__)


class SnapshotCollector:
    """Polls Frigate API for new detection events, downloads clean snapshots and crops."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.client = FrigateClient(
            os.environ.get("FRIGATE_URL", self.config["frigate"]["url"])
        )
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.crop_dir = self.data_dir / "crops"
        self.poll_interval = self.config["collection"]["poll_interval_minutes"] * 60
        self.batch_trigger = self.config["collection"]["batch_trigger_size"]
        self.dedup_window = self.config["collection"]["dedup_window_seconds"]

        # Map COCO labels to custom classes
        self.class_map = self.config["classes"]["coco_to_custom"]
        # All COCO labels we care about
        self.tracked_labels = list(self.class_map.keys())

        init_db()

    def collect_once(self) -> int:
        """Run one collection cycle. Returns number of new events saved."""
        session = get_session()
        try:
            # Find the most recent event we've collected
            last_event = (
                session.query(Event)
                .order_by(Event.start_time.desc())
                .first()
            )
            after = last_event.start_time.timestamp() if last_event else None

            events = self.client.get_events(
                after=after,
                labels=self.tracked_labels,
                has_snapshot=True,
                limit=500,
            )

            saved = 0
            for event in events:
                # Skip if already collected
                if session.query(Event).filter_by(id=event.id).first():
                    continue

                try:
                    saved += self._save_event(session, event)
                except Exception:
                    logger.exception(f"Failed to save event {event.id}")

            session.commit()

            if saved > 0:
                logger.info(f"Collected {saved} new events")
                self._check_batch_trigger(session)

            return saved
        finally:
            session.close()

    def _save_event(self, session, event) -> int:
        """Download snapshot, crop detection, and save to database."""
        # Download clean snapshot
        try:
            snapshot_bytes = self.client.download_snapshot_clean(event.id)
        except Exception:
            logger.warning(f"No clean snapshot for {event.id}, trying annotated")
            snapshot_bytes = self.client.download_snapshot(event.id)

        img = Image.open(io.BytesIO(snapshot_bytes))
        img_w, img_h = img.size

        # Save full-frame snapshot
        date_str = event.start_time.strftime("%Y-%m-%d")
        camera_dir = self.raw_dir / event.camera / date_str
        camera_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = camera_dir / f"{event.id}.png"
        img.save(snapshot_path, "PNG")

        # Crop the detection region
        x, y, w, h = event.box
        # Convert normalized coordinates to pixels
        px_x = int(x * img_w)
        px_y = int(y * img_h)
        px_w = int(w * img_w)
        px_h = int(h * img_h)
        # Add padding (10% on each side)
        pad_x = int(px_w * 0.1)
        pad_y = int(px_h * 0.1)
        crop_box = (
            max(0, px_x - pad_x),
            max(0, px_y - pad_y),
            min(img_w, px_x + px_w + pad_x),
            min(img_h, px_y + px_h + pad_y),
        )
        crop = img.crop(crop_box)

        crop_camera_dir = self.crop_dir / event.camera / date_str
        crop_camera_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_camera_dir / f"{event.id}.png"
        crop.save(crop_path, "PNG")

        # Save to database
        db_event = Event(
            id=event.id,
            camera=event.camera,
            frigate_label=event.label,
            frigate_score=event.score,
            box_x=event.box[0],
            box_y=event.box[1],
            box_w=event.box[2],
            box_h=event.box[3],
            snapshot_path=str(snapshot_path),
            crop_path=str(crop_path),
            start_time=event.start_time,
        )
        session.add(db_event)
        return 1

    def _check_batch_trigger(self, session):
        """Check if enough unclassified events exist to trigger a batch."""
        unclassified = (
            session.query(Event)
            .filter_by(classified=False, batch_id=None)
            .count()
        )
        if unclassified >= self.batch_trigger:
            batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            batch = Batch(id=batch_id, event_count=unclassified)
            session.add(batch)

            # Assign events to batch
            events = (
                session.query(Event)
                .filter_by(classified=False, batch_id=None)
                .limit(self.config["collection"]["max_batch_size"])
                .all()
            )
            for event in events:
                event.batch_id = batch_id
            batch.event_count = len(events)
            session.commit()

            logger.info(
                f"Created batch {batch_id} with {len(events)} events — ready for classification"
            )

    def run(self):
        """Run the collector daemon continuously."""
        logger.info(
            f"Starting snapshot collector (polling every {self.poll_interval}s)"
        )

        if not self.client.health_check():
            logger.error(
                f"Cannot reach Frigate at {self.client.base_url}. Check FRIGATE_URL."
            )
            return

        logger.info(f"Connected to Frigate at {self.client.base_url}")

        while True:
            try:
                self.collect_once()
            except Exception:
                logger.exception("Collection cycle failed")
            time.sleep(self.poll_interval)
