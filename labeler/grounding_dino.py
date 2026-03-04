"""Grounding DINO package discovery — finds packages in full-frame snapshots."""

import logging
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

from collector.db import Event, get_session, init_db
from collector.frigate_client import FrigateClient

logger = logging.getLogger(__name__)


class PackageDiscovery:
    """Scans full-frame snapshots with Grounding DINO to find packages.

    Since Frigate's COCO model doesn't detect packages, this runs
    separately to discover them in scenes from porch/doorstep cameras.
    """

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        pd_cfg = self.config["package_discovery"]
        self.cameras = pd_cfg["cameras"]
        self.prompts = pd_cfg["grounding_dino"]["prompts"]
        self.confidence_threshold = pd_cfg["grounding_dino"]["confidence_threshold"]

        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.crop_dir = self.data_dir / "crops"

        self._model = None

    @property
    def model(self):
        """Lazy-load Grounding DINO model."""
        if self._model is None:
            try:
                from autodistill_grounding_dino import GroundingDINO
                from autodistill.detection import CaptionOntology

                # Build ontology mapping all prompts to "package"
                ontology = CaptionOntology(
                    {prompt: "package" for prompt in self.prompts}
                )
                self._model = GroundingDINO(
                    ontology=ontology,
                    box_threshold=self.confidence_threshold,
                    text_threshold=self.confidence_threshold,
                )
                logger.info("Grounding DINO loaded for package discovery")
            except ImportError:
                logger.error(
                    "autodistill-grounding-dino not installed. "
                    "Install with: pip install autodistill-grounding-dino"
                )
                raise
        return self._model

    def scan_cameras(self) -> int:
        """Grab current snapshots from package-relevant cameras and scan for packages.

        Returns number of packages discovered.
        """
        import os

        client = FrigateClient(
            os.environ.get("FRIGATE_URL", self.config["frigate"]["url"])
        )

        init_db()
        session = get_session()
        discovered = 0

        try:
            for camera in self.cameras:
                try:
                    snapshot_bytes = client.get_camera_snapshot(camera)
                    count = self._process_snapshot(
                        session, camera, snapshot_bytes
                    )
                    discovered += count
                except Exception:
                    logger.exception(f"Failed to scan camera {camera}")

            session.commit()
            if discovered > 0:
                logger.info(f"Discovered {discovered} packages across {len(self.cameras)} cameras")
            return discovered
        finally:
            session.close()

    def scan_directory(self, directory: str | Path) -> int:
        """Scan manually uploaded images for packages.

        Args:
            directory: Path to directory containing images.

        Returns number of packages discovered.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return 0

        init_db()
        session = get_session()
        discovered = 0

        try:
            for img_path in sorted(directory.glob("*")):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    count = self._process_snapshot(
                        session, "manual_upload", img_bytes, source_path=str(img_path)
                    )
                    discovered += count
                except Exception:
                    logger.exception(f"Failed to process {img_path}")

            session.commit()
            if discovered > 0:
                logger.info(f"Discovered {discovered} packages in {directory}")
            return discovered
        finally:
            session.close()

    def _process_snapshot(
        self,
        session,
        camera: str,
        snapshot_bytes: bytes,
        source_path: str | None = None,
    ) -> int:
        """Run Grounding DINO on a snapshot, save any package detections."""
        import io
        import numpy as np

        img = Image.open(io.BytesIO(snapshot_bytes))
        img_w, img_h = img.size

        # Save the full frame temporarily for Grounding DINO
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, "PNG")
            tmp_path = tmp.name

        try:
            # Run Grounding DINO
            detections = self.model.predict(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if len(detections) == 0:
            return 0

        count = 0
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2] in pixels
            confidence = float(detections.confidence[i])

            if confidence < self.confidence_threshold:
                continue

            # Convert to normalized [x, y, w, h]
            x1, y1, x2, y2 = bbox
            norm_x = float(x1) / img_w
            norm_y = float(y1) / img_h
            norm_w = float(x2 - x1) / img_w
            norm_h = float(y2 - y1) / img_h

            # Generate unique event ID
            event_id = f"pkg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Save full snapshot
            camera_dir = self.raw_dir / camera / date_str
            camera_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = camera_dir / f"{event_id}.png"
            img.save(snapshot_path, "PNG")

            # Save crop
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            crop_box = (
                max(0, int(x1) - pad_x),
                max(0, int(y1) - pad_y),
                min(img_w, int(x2) + pad_x),
                min(img_h, int(y2) + pad_y),
            )
            crop = img.crop(crop_box)
            crop_camera_dir = self.crop_dir / camera / date_str
            crop_camera_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_camera_dir / f"{event_id}.png"
            crop.save(crop_path, "PNG")

            # Save to database
            db_event = Event(
                id=event_id,
                camera=camera,
                frigate_label="package",
                frigate_score=confidence,
                box_x=norm_x,
                box_y=norm_y,
                box_w=norm_w,
                box_h=norm_h,
                snapshot_path=str(snapshot_path),
                crop_path=str(crop_path),
                start_time=datetime.utcnow(),
            )
            session.add(db_event)
            count += 1

        return count
