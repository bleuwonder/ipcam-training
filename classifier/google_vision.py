"""Google Cloud Vision API fallback classifier."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default label mapping from Google Vision labels to our classes
DEFAULT_LABEL_MAP = {
    # Human
    "Person": "human",
    "Human": "human",
    "Man": "human",
    "Woman": "human",
    "Child": "human",
    "People": "human",
    "Pedestrian": "human",
    # Vehicle
    "Car": "vehicle",
    "Truck": "vehicle",
    "Vehicle": "vehicle",
    "Motor vehicle": "vehicle",
    "Motorcycle": "vehicle",
    "Bicycle": "vehicle",
    "Bus": "vehicle",
    "Van": "vehicle",
    "Automotive": "vehicle",
    # Animal
    "Dog": "animal",
    "Cat": "animal",
    "Animal": "animal",
    "Bird": "animal",
    "Deer": "animal",
    "Mammal": "animal",
    "Pet": "animal",
    "Wildlife": "animal",
    # Package
    "Package delivery": "package",
    "Cardboard": "package",
    "Box": "package",
    "Shipping box": "package",
    "Parcel": "package",
    "Carton": "package",
    "Package": "package",
}


@dataclass
class GoogleVisionResult:
    """Result from Google Cloud Vision API."""

    predicted_class: str | None
    confidence: float
    raw_labels: list[tuple[str, float]]  # (label, score) pairs


class GoogleVisionClassifier:
    """Fallback classifier using Google Cloud Vision API.

    Used when CLIP is uncertain (confidence 40-80%).
    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable.
    """

    def __init__(self, label_mapping: dict[str, str] | None = None):
        self.label_map = label_mapping or DEFAULT_LABEL_MAP
        self._client = None

    @property
    def client(self):
        """Lazy-load the Vision API client."""
        if self._client is None:
            try:
                from google.cloud import vision

                self._client = vision.ImageAnnotatorClient()
                logger.info("Google Cloud Vision API client initialized")
            except Exception:
                logger.error(
                    "Failed to initialize Google Vision client. "
                    "Ensure google-cloud-vision is installed and "
                    "GOOGLE_APPLICATION_CREDENTIALS is set."
                )
                raise
        return self._client

    def classify(self, image_path: str | Path) -> GoogleVisionResult:
        """Classify an image using Google Cloud Vision API.

        Args:
            image_path: Path to the image file (crop).

        Returns:
            GoogleVisionResult with predicted class and confidence.
        """
        from google.cloud import vision

        with open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)
        response = self.client.label_detection(image=image, max_results=10)

        if response.error.message:
            raise RuntimeError(
                f"Google Vision API error: {response.error.message}"
            )

        raw_labels = [
            (label.description, label.score) for label in response.label_annotations
        ]

        # Map Google labels to our classes
        best_class = None
        best_score = 0.0

        for label_desc, score in raw_labels:
            mapped = self.label_map.get(label_desc)
            if mapped and score > best_score:
                best_class = mapped
                best_score = score

        return GoogleVisionResult(
            predicted_class=best_class,
            confidence=best_score,
            raw_labels=raw_labels,
        )

    def is_available(self) -> bool:
        """Check if Google Vision API is configured and reachable."""
        try:
            _ = self.client
            return True
        except Exception:
            return False
