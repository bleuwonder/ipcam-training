"""CLIP zero-shot classifier for re-classifying Frigate detection crops."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import open_clip
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result of CLIP classification on a crop."""

    predicted_class: str
    confidence: float
    all_scores: dict[str, float]  # class -> score


class ClipClassifier:
    """Uses OpenCLIP ViT-L/14 for zero-shot classification of detection crops.

    Takes a cropped image from a Frigate detection and classifies it
    against the target classes using text prompts.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str | None = None,
        prompts: dict[str, list[str]] | None = None,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP {model_name} on {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Default prompts per class
        self.prompts = prompts or {
            "human": [
                "a photo of a person",
                "a photo of a human",
                "a photo of someone walking",
            ],
            "vehicle": [
                "a photo of a car",
                "a photo of a truck",
                "a photo of a van",
                "a photo of a motorcycle",
                "a photo of a bicycle",
            ],
            "animal": [
                "a photo of a dog",
                "a photo of a cat",
                "a photo of a deer",
                "a photo of a bird",
                "a photo of a raccoon",
                "a photo of a squirrel",
            ],
            "package": [
                "a photo of a delivery package",
                "a photo of a cardboard box",
                "a photo of a parcel",
            ],
            "false_positive": [
                "a photo of nothing",
                "a photo of an empty scene",
                "a photo of a shadow",
            ],
        }

        # Pre-encode all text prompts
        self._encode_prompts()
        logger.info(f"CLIP classifier ready ({len(self.class_features)} classes)")

    def _encode_prompts(self):
        """Pre-encode text prompts for each class."""
        self.class_features = {}
        with torch.no_grad():
            for class_name, texts in self.prompts.items():
                tokens = self.tokenizer(texts).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                # Average the prompt embeddings for this class
                self.class_features[class_name] = features.mean(dim=0, keepdim=True)
                self.class_features[class_name] /= self.class_features[
                    class_name
                ].norm(dim=-1, keepdim=True)

    def classify(self, image_path: str | Path) -> ClipResult:
        """Classify a crop image against all target classes.

        Args:
            image_path: Path to the cropped detection image.

        Returns:
            ClipResult with predicted class, confidence, and all scores.
        """
        img = Image.open(image_path).convert("RGB")
        return self.classify_image(img)

    def classify_image(self, img: Image.Image) -> ClipResult:
        """Classify a PIL Image against all target classes."""
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity to each class
            scores = {}
            for class_name, text_features in self.class_features.items():
                similarity = (image_features @ text_features.T).squeeze().item()
                # Convert cosine similarity to a 0-1 range
                # CLIP similarities are typically in [-1, 1] but usually [0.15, 0.40]
                scores[class_name] = similarity

        # Softmax over similarities to get proper probabilities
        import torch.nn.functional as F

        score_tensor = torch.tensor(list(scores.values()))
        # Temperature scaling — lower temp = more peaked distribution
        probs = F.softmax(score_tensor * 100, dim=0)
        prob_scores = {
            name: prob.item() for name, prob in zip(scores.keys(), probs)
        }

        # Best class (excluding false_positive unless it's dominant)
        best_class = max(prob_scores, key=prob_scores.get)
        best_score = prob_scores[best_class]

        return ClipResult(
            predicted_class=best_class,
            confidence=best_score,
            all_scores=prob_scores,
        )

    def classify_batch(self, image_paths: list[str | Path]) -> list[ClipResult]:
        """Classify multiple crops efficiently."""
        results = []
        # Process in mini-batches to manage VRAM
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(self.preprocess(img))

            image_batch = torch.stack(images).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                for j in range(len(batch_paths)):
                    feat = image_features[j : j + 1]
                    scores = {}
                    for class_name, text_features in self.class_features.items():
                        similarity = (feat @ text_features.T).squeeze().item()
                        scores[class_name] = similarity

                    import torch.nn.functional as F

                    score_tensor = torch.tensor(list(scores.values()))
                    probs = F.softmax(score_tensor * 100, dim=0)
                    prob_scores = {
                        name: prob.item()
                        for name, prob in zip(scores.keys(), probs)
                    }

                    best_class = max(prob_scores, key=prob_scores.get)
                    best_score = prob_scores[best_class]

                    results.append(
                        ClipResult(
                            predicted_class=best_class,
                            confidence=best_score,
                            all_scores=prob_scores,
                        )
                    )

        return results
