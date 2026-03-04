"""Export trained YOLO model to ONNX format for Frigate."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports a trained YOLO .pt model to ONNX for Frigate's yolo-generic detector."""

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        t_cfg = self.config["training"]
        self.export_imgsz = t_cfg["export_imgsz"]
        self.classes = self.config["classes"]["custom"]

    def export(self, model_path: str | Path) -> Path:
        """Export a .pt model to ONNX.

        Args:
            model_path: Path to best.pt weights file.

        Returns:
            Path to the exported ONNX file.
        """
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(
            f"Exporting {model_path.name} to ONNX "
            f"(imgsz={self.export_imgsz})..."
        )

        model = YOLO(str(model_path))
        export_path = model.export(
            format="onnx",
            imgsz=self.export_imgsz,
            simplify=True,
            dynamic=False,    # Fixed input size for Frigate
            opset=12,         # Broad compatibility
        )

        onnx_path = Path(export_path)
        logger.info(f"ONNX model saved: {onnx_path}")

        # Validate the exported model
        self._validate_onnx(onnx_path)

        return onnx_path

    def _validate_onnx(self, onnx_path: Path):
        """Run a quick sanity check on the exported ONNX model."""
        try:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path))
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape

            logger.info(f"ONNX validation: input={input_name}, shape={input_shape}")

            # Create a dummy input
            dummy = np.random.randn(
                1, 3, self.export_imgsz, self.export_imgsz
            ).astype(np.float32)
            outputs = sess.run(None, {input_name: dummy})
            logger.info(
                f"ONNX validation passed — output shape: {outputs[0].shape}"
            )
        except ImportError:
            logger.warning("onnxruntime not installed — skipping validation")
        except Exception:
            logger.exception("ONNX validation failed")

    def write_labelmap(self, output_path: str | Path) -> Path:
        """Write Frigate-compatible labelmap.txt.

        Frigate expects one label per line, index 0 first.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(self.classes) + "\n")
        logger.info(f"Labelmap written: {output_path}")
        return output_path
