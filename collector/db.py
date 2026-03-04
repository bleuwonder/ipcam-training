"""SQLite database models for tracking events, classifications, and training runs."""

import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class Event(Base):
    """A Frigate detection event with snapshot and bounding box."""

    __tablename__ = "events"

    id = Column(String, primary_key=True)  # Frigate event ID
    camera = Column(String, nullable=False, index=True)
    frigate_label = Column(String, nullable=False, index=True)
    frigate_score = Column(Float, nullable=False)
    # Bounding box (normalized 0-1 coordinates from Frigate)
    box_x = Column(Float)
    box_y = Column(Float)
    box_w = Column(Float)
    box_h = Column(Float)
    # File paths
    snapshot_path = Column(String)  # Full-frame clean snapshot
    crop_path = Column(String)  # Cropped detection region
    # Timestamps
    start_time = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow)
    # Processing state
    classified = Column(Boolean, default=False, index=True)
    batch_id = Column(String, index=True)


class Classification(Base):
    """Result of CLIP/Google Vision classification on an event crop."""

    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, nullable=False, index=True)
    # CLIP results
    clip_label = Column(String)
    clip_score = Column(Float)
    # Google Vision results (optional fallback)
    google_label = Column(String)
    google_score = Column(Float)
    # Final decision
    final_label = Column(String, nullable=False)
    decision = Column(
        String, nullable=False
    )  # auto_approved, auto_corrected, flagged_review
    # Whether Frigate and classifier agreed
    agrees_with_frigate = Column(Boolean)
    # Human review outcome (filled in after Label Studio export)
    human_label = Column(String)
    human_reviewed = Column(Boolean, default=False)
    # Approved for training?
    approved = Column(Boolean, default=False, index=True)
    approved_at = Column(DateTime)
    classified_at = Column(DateTime, default=datetime.utcnow)


class Batch(Base):
    """A batch of events grouped for processing."""

    __tablename__ = "batches"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    event_count = Column(Integer, default=0)
    status = Column(
        String, default="pending"
    )  # pending, classifying, reviewing, approved, training
    label_studio_project_id = Column(Integer)


class TrainingRun(Base):
    """Record of a YOLO training run."""

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    # Dataset info
    total_images = Column(Integer)
    images_per_class = Column(Text)  # JSON string
    # Training config
    model_base = Column(String)
    epochs_completed = Column(Integer)
    batch_size = Column(Integer)
    # Metrics
    map50 = Column(Float)
    map50_95 = Column(Float)
    per_class_map50 = Column(Text)  # JSON string
    # Output
    model_path = Column(String)
    export_path = Column(String)
    passed_quality_gates = Column(Boolean)
    deployed = Column(Boolean, default=False)
    version = Column(String)


def get_engine(db_url: str | None = None):
    """Create database engine."""
    url = db_url or os.environ.get("DATABASE_URL", "sqlite:///pipeline.db")
    return create_engine(url, echo=False)


def get_session(db_url: str | None = None) -> Session:
    """Create a new database session."""
    engine = get_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db(db_url: str | None = None):
    """Create all tables."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
