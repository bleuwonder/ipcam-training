"""Deploy exported model to Frigate config directory."""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from collector.db import TrainingRun, get_session
from deployer.export import ModelExporter

logger = logging.getLogger(__name__)


class ModelDeployer:
    """Copies the exported ONNX model into Frigate's config directory."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        d_cfg = self.config["deployment"]
        self.frigate_model_dir = Path(
            os.environ.get("FRIGATE_MODEL_DIR", d_cfg["frigate_model_dir"])
        )
        self.keep_versions = d_cfg["keep_versions"]
        self.models_dir = Path("models")
        self.exporter = ModelExporter(config_path)

    def deploy(self, model_path: str | Path, version: str | None = None) -> Path:
        """Export and deploy a model to Frigate.

        Args:
            model_path: Path to the .pt weights file.
            version: Version tag (e.g. "v003_20260304"). Auto-detected if None.

        Returns:
            Path to the deployed ONNX file.
        """
        model_path = Path(model_path)

        # Determine version from parent directory name if not given
        if version is None:
            version = model_path.parent.name

        # Export to ONNX (saves next to the .pt file)
        onnx_path = self.exporter.export(model_path)

        # Write labelmap.txt next to the model
        labelmap_src = model_path.parent / "labelmap.txt"
        self.exporter.write_labelmap(labelmap_src)

        # Archive current model before overwriting
        self._archive_current(version)

        # Copy new model to Frigate config dir
        self.frigate_model_dir.mkdir(parents=True, exist_ok=True)
        dst_onnx = self.frigate_model_dir / "best.onnx"
        dst_labelmap = self.frigate_model_dir / "labelmap.txt"

        shutil.copy2(onnx_path, dst_onnx)
        shutil.copy2(labelmap_src, dst_labelmap)
        logger.info(f"Deployed model to {self.frigate_model_dir}")

        # Update database record
        session = get_session()
        try:
            run = (
                session.query(TrainingRun)
                .filter_by(version=version)
                .first()
            )
            if run:
                run.export_path = str(dst_onnx)
                run.deployed = True
                session.commit()
        finally:
            session.close()

        # Prune old model archives
        self._prune_old_versions()

        # Print Frigate config snippet
        self._print_frigate_snippet()

        return dst_onnx

    def _archive_current(self, new_version: str):
        """Archive whatever model is currently deployed."""
        dst_onnx = self.frigate_model_dir / "best.onnx"
        if not dst_onnx.exists():
            return

        archive_dir = self.frigate_model_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"best_{ts}.onnx"
        shutil.copy2(dst_onnx, archive_path)
        logger.info(f"Archived current model to {archive_path}")

    def _prune_old_versions(self):
        """Remove model archives beyond keep_versions limit."""
        archive_dir = self.frigate_model_dir / "archive"
        if not archive_dir.exists():
            return

        archives = sorted(archive_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        excess = len(archives) - self.keep_versions
        if excess > 0:
            for old in archives[:excess]:
                old.unlink()
                logger.info(f"Pruned old model archive: {old.name}")

    def rollback(self):
        """Restore the most recent archived model."""
        archive_dir = self.frigate_model_dir / "archive"
        if not archive_dir.exists():
            logger.error("No archive directory found — nothing to roll back to")
            return False

        archives = sorted(archive_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if not archives:
            logger.error("No archived models found")
            return False

        latest_archive = archives[-1]
        dst = self.frigate_model_dir / "best.onnx"
        shutil.copy2(latest_archive, dst)
        latest_archive.unlink()  # Remove from archive since it's now active
        logger.info(f"Rolled back to {latest_archive.name}")
        return True

    def _print_frigate_snippet(self):
        """Print the Frigate config YAML snippet the user needs to apply."""
        model_path_in_container = "/config/models/best.onnx"
        labelmap_path_in_container = "/config/models/labelmap.txt"

        snippet = f"""
╔══════════════════════════════════════════════════════════════╗
║             FRIGATE CONFIG — APPLY THESE CHANGES             ║
╚══════════════════════════════════════════════════════════════╝

Add or update these sections in your Frigate config.yml:

  detectors:
    onnx:
      type: onnx

  model:
    model_type: yolo-generic
    width: {self.config['training']['export_imgsz']}
    height: {self.config['training']['export_imgsz']}
    input_tensor: nchw
    input_dtype: float
    path: {model_path_in_container}
    labelmap_path: {labelmap_path_in_container}

  objects:
    track:
      - human
      - package
      - animal
      - vehicle

Then restart Frigate to load the new model.
"""
        print(snippet)
