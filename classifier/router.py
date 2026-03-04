"""Classification router — decides auto-approve, auto-correct, or flag for review."""

import logging
from datetime import datetime

import yaml

from classifier.clip_classifier import ClipClassifier
from classifier.google_vision import GoogleVisionClassifier
from collector.db import Classification, Event, get_session

logger = logging.getLogger(__name__)


class ClassificationRouter:
    """Routes detection crops through CLIP → Google Vision → human review.

    Decision logic:
    - CLIP >80% & agrees with Frigate → auto-approve
    - CLIP >80% & disagrees → auto-correct
    - CLIP 40-80% → Google Vision second opinion
      - Google agrees with CLIP >80% → auto-approve
      - Otherwise → flag for human review
    - CLIP <40% → flag for human review
    """

    def __init__(self, config_path: str = "config/pipeline.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        clip_cfg = self.config["classifier"]["clip"]
        self.auto_approve_threshold = clip_cfg["auto_approve_threshold"]
        self.uncertain_threshold = clip_cfg["uncertain_threshold"]

        # Class mapping from COCO labels to custom classes
        self.class_map = self.config["classes"]["coco_to_custom"]

        # Initialize CLIP classifier
        self.clip = ClipClassifier(
            model_name=clip_cfg["model"],
            pretrained=clip_cfg["pretrained"],
            device=clip_cfg.get("device"),
            prompts=clip_cfg.get("prompts"),
        )

        # Initialize Google Vision (optional)
        gv_cfg = self.config["classifier"]["google_vision"]
        self.google_vision = None
        if gv_cfg.get("enabled"):
            try:
                self.google_vision = GoogleVisionClassifier(
                    label_mapping=gv_cfg.get("label_mapping")
                )
                if self.google_vision.is_available():
                    logger.info("Google Vision API fallback enabled")
                else:
                    logger.warning(
                        "Google Vision API not available — running CLIP-only"
                    )
                    self.google_vision = None
            except Exception:
                logger.warning("Google Vision API init failed — running CLIP-only")
                self.google_vision = None

    def classify_event(self, event: Event) -> Classification:
        """Classify a single event's crop and decide routing.

        Args:
            event: Database Event with crop_path set.

        Returns:
            Classification record with decision.
        """
        # Map Frigate's COCO label to our custom class
        frigate_custom_label = self.class_map.get(event.frigate_label)

        # Run CLIP on the crop
        clip_result = self.clip.classify(event.crop_path)
        clip_label = clip_result.predicted_class
        clip_score = clip_result.confidence

        # Skip false_positive class for label comparison
        if clip_label == "false_positive":
            # If CLIP thinks it's a false positive, always send to human review
            return self._make_classification(
                event_id=event.id,
                clip_label="false_positive",
                clip_score=clip_score,
                final_label=frigate_custom_label or event.frigate_label,
                decision="flagged_review",
                agrees=False,
            )

        google_label = None
        google_score = None

        if clip_score >= self.auto_approve_threshold:
            # CLIP is confident
            agrees = clip_label == frigate_custom_label
            if agrees:
                decision = "auto_approved"
                final_label = clip_label
            else:
                decision = "auto_corrected"
                final_label = clip_label
                logger.info(
                    f"Event {event.id}: CLIP corrected {event.frigate_label} → {clip_label} "
                    f"(confidence {clip_score:.2f})"
                )
        elif clip_score >= self.uncertain_threshold:
            # CLIP is uncertain — try Google Vision
            if self.google_vision:
                try:
                    gv_result = self.google_vision.classify(event.crop_path)
                    google_label = gv_result.predicted_class
                    google_score = gv_result.confidence

                    if google_label and google_label == clip_label and google_score > 0.80:
                        decision = "auto_approved"
                        final_label = clip_label
                        agrees = clip_label == frigate_custom_label
                    else:
                        decision = "flagged_review"
                        final_label = clip_label  # Suggest CLIP's label
                        agrees = clip_label == frigate_custom_label
                except Exception:
                    logger.exception("Google Vision failed, flagging for review")
                    decision = "flagged_review"
                    final_label = clip_label
                    agrees = clip_label == frigate_custom_label
            else:
                # No Google Vision, flag for review
                decision = "flagged_review"
                final_label = clip_label
                agrees = clip_label == frigate_custom_label
        else:
            # CLIP very uncertain
            decision = "flagged_review"
            final_label = frigate_custom_label or clip_label
            agrees = clip_label == frigate_custom_label

        return self._make_classification(
            event_id=event.id,
            clip_label=clip_label,
            clip_score=clip_score,
            google_label=google_label,
            google_score=google_score,
            final_label=final_label,
            decision=decision,
            agrees=agrees,
        )

    def _make_classification(
        self,
        event_id: str,
        clip_label: str,
        clip_score: float,
        final_label: str,
        decision: str,
        agrees: bool,
        google_label: str | None = None,
        google_score: float | None = None,
    ) -> Classification:
        is_approved = decision in ("auto_approved", "auto_corrected")
        return Classification(
            event_id=event_id,
            clip_label=clip_label,
            clip_score=clip_score,
            google_label=google_label,
            google_score=google_score,
            final_label=final_label,
            decision=decision,
            agrees_with_frigate=agrees,
            approved=is_approved,
            approved_at=datetime.utcnow() if is_approved else None,
        )

    def classify_batch(self, batch_id: str | None = None):
        """Classify all unclassified events (optionally filtered by batch).

        Returns dict with counts: auto_approved, auto_corrected, flagged_review.
        """
        session = get_session()
        try:
            query = session.query(Event).filter_by(classified=False)
            if batch_id:
                query = query.filter_by(batch_id=batch_id)

            events = query.all()
            if not events:
                logger.info("No unclassified events found")
                return {"auto_approved": 0, "auto_corrected": 0, "flagged_review": 0}

            counts = {"auto_approved": 0, "auto_corrected": 0, "flagged_review": 0}

            for event in events:
                if not event.crop_path:
                    logger.warning(f"Event {event.id} has no crop — skipping")
                    continue

                classification = self.classify_event(event)
                session.add(classification)
                event.classified = True
                counts[classification.decision] = (
                    counts.get(classification.decision, 0) + 1
                )

            session.commit()
            logger.info(
                f"Classified {len(events)} events: "
                f"{counts['auto_approved']} auto-approved, "
                f"{counts['auto_corrected']} auto-corrected, "
                f"{counts['flagged_review']} flagged for review"
            )
            return counts
        finally:
            session.close()
