#!/usr/bin/env python3
"""Frigate Training Pipeline CLI.

Usage:
    python cli.py collect start         # Start collection daemon
    python cli.py collect run-once      # Run one collection cycle
    python cli.py collect status        # Show per-camera/class stats

    python cli.py classify run          # Run CLIP on all unclassified crops
    python cli.py classify stats        # Auto-approved vs flagged counts

    python cli.py review import         # Push flagged items to Label Studio
    python cli.py review export         # Pull approved annotations from Label Studio

    python cli.py discover run          # Run Grounding DINO package scan
    python cli.py discover upload DIR   # Scan a directory of manually uploaded images

    python cli.py train build           # Build training dataset from approved data
    python cli.py train run             # Train YOLO11m
    python cli.py train evaluate MODEL  # Evaluate a model against the dataset

    python cli.py deploy export MODEL   # Export .pt → ONNX
    python cli.py deploy install MODEL  # Export and copy to Frigate config dir
    python cli.py deploy rollback       # Revert to previous model

    python cli.py pipeline status       # Show overall pipeline dashboard
"""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    default="config/pipeline.yaml",
    envvar="PIPELINE_CONFIG",
    show_default=True,
    help="Path to pipeline config file.",
)
@click.pass_context
def cli(ctx, config):
    """Frigate Camera Training Pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


# ──────────────────────────────────────────────────────────
# COLLECT
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def collect(ctx):
    """Collect detection snapshots from Frigate."""


@collect.command("start")
@click.pass_context
def collect_start(ctx):
    """Start the collection daemon (runs continuously)."""
    from collector.snapshot_collector import SnapshotCollector

    click.echo("Starting snapshot collector daemon... (Ctrl+C to stop)")
    collector = SnapshotCollector(ctx.obj["config"])
    collector.run()


@collect.command("run-once")
@click.pass_context
def collect_once(ctx):
    """Run a single collection cycle."""
    from collector.snapshot_collector import SnapshotCollector

    collector = SnapshotCollector(ctx.obj["config"])
    saved = collector.collect_once()
    click.echo(f"Collected {saved} new events.")


@collect.command("status")
@click.pass_context
def collect_status(ctx):
    """Show collection statistics."""
    from collector.db import Batch, Classification, Event, get_session

    session = get_session()
    try:
        total = session.query(Event).count()
        unclassified = session.query(Event).filter_by(classified=False).count()
        approved = session.query(Classification).filter_by(approved=True).count()
        pending_review = (
            session.query(Classification)
            .filter_by(decision="flagged_review", human_reviewed=False)
            .count()
        )
        batches = session.query(Batch).count()

        click.echo(f"\n{'═' * 50}")
        click.echo("  Collection Status")
        click.echo(f"{'═' * 50}")
        click.echo(f"  Total events collected:  {total}")
        click.echo(f"  Unclassified:            {unclassified}")
        click.echo(f"  Approved for training:   {approved}")
        click.echo(f"  Pending human review:    {pending_review}")
        click.echo(f"  Total batches:           {batches}")

        # Per-camera breakdown
        from sqlalchemy import func
        camera_counts = (
            session.query(Event.camera, func.count(Event.id))
            .group_by(Event.camera)
            .all()
        )
        if camera_counts:
            click.echo(f"\n  Per-camera breakdown:")
            for camera, count in sorted(camera_counts):
                click.echo(f"    {camera:20s}: {count}")

        # Per-class approved breakdown
        class_counts = (
            session.query(Classification.final_label, func.count(Classification.id))
            .filter_by(approved=True)
            .group_by(Classification.final_label)
            .all()
        )
        if class_counts:
            click.echo(f"\n  Approved per class:")
            for label, count in sorted(class_counts):
                click.echo(f"    {label:20s}: {count}")

        click.echo(f"{'═' * 50}\n")
    finally:
        session.close()


# ──────────────────────────────────────────────────────────
# CLASSIFY
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def classify(ctx):
    """Run CLIP classifier on collected crops."""


@classify.command("run")
@click.option("--batch-id", default=None, help="Process specific batch only.")
@click.pass_context
def classify_run(ctx, batch_id):
    """Run CLIP (+ optional Google Vision) on unclassified crops."""
    from classifier.router import ClassificationRouter

    click.echo("Running classification...")
    router = ClassificationRouter(ctx.obj["config"])
    counts = router.classify_batch(batch_id=batch_id)
    click.echo(
        f"Done. Auto-approved: {counts['auto_approved']}, "
        f"Auto-corrected: {counts['auto_corrected']}, "
        f"Flagged for review: {counts['flagged_review']}"
    )


@classify.command("stats")
@click.pass_context
def classify_stats(ctx):
    """Show classification statistics."""
    from collector.db import Classification, get_session
    from sqlalchemy import func

    session = get_session()
    try:
        rows = (
            session.query(Classification.decision, func.count(Classification.id))
            .group_by(Classification.decision)
            .all()
        )
        click.echo(f"\n{'═' * 50}")
        click.echo("  Classification Stats")
        click.echo(f"{'═' * 50}")
        for decision, count in rows:
            click.echo(f"  {decision:25s}: {count}")

        # Show per-class approved breakdown
        class_rows = (
            session.query(Classification.final_label, func.count(Classification.id))
            .filter_by(approved=True)
            .group_by(Classification.final_label)
            .all()
        )
        if class_rows:
            click.echo(f"\n  Approved per class:")
            for label, count in sorted(class_rows):
                click.echo(f"    {label:20s}: {count}")
        click.echo(f"{'═' * 50}\n")
    finally:
        session.close()


# ──────────────────────────────────────────────────────────
# REVIEW
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def review(ctx):
    """Label Studio human review workflow."""


@review.command("import")
@click.option("--batch-id", default=None, help="Import specific batch only.")
@click.pass_context
def review_import(ctx, batch_id):
    """Push flagged detections to Label Studio for human review."""
    from labeler.label_studio_import import LabelStudioImporter

    click.echo("Importing flagged events to Label Studio...")
    importer = LabelStudioImporter(ctx.obj["config"])
    count = importer.import_flagged_events(batch_id=batch_id)
    if count > 0:
        ls_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
        click.echo(
            f"Imported {count} tasks. Open {ls_url} to start reviewing."
        )
    else:
        click.echo("Nothing to import.")


@review.command("export")
@click.pass_context
def review_export(ctx):
    """Pull completed annotations from Label Studio."""
    from labeler.label_studio_export import LabelStudioExporter

    click.echo("Exporting approved annotations from Label Studio...")
    exporter = LabelStudioExporter(ctx.obj["config"])
    count = exporter.export_approved()
    click.echo(f"Exported {count} annotations.")


# ──────────────────────────────────────────────────────────
# DISCOVER
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def discover(ctx):
    """Grounding DINO package discovery."""


@discover.command("run")
@click.pass_context
def discover_run(ctx):
    """Scan porch/doorstep cameras for packages with Grounding DINO."""
    from labeler.grounding_dino import PackageDiscovery

    click.echo("Running Grounding DINO package scan...")
    scanner = PackageDiscovery(ctx.obj["config"])
    count = scanner.scan_cameras()
    click.echo(f"Discovered {count} packages.")


@discover.command("upload")
@click.argument("directory")
@click.pass_context
def discover_upload(ctx, directory):
    """Scan a directory of manually uploaded images for packages."""
    from labeler.grounding_dino import PackageDiscovery

    click.echo(f"Scanning {directory} for packages...")
    scanner = PackageDiscovery(ctx.obj["config"])
    count = scanner.scan_directory(directory)
    click.echo(f"Discovered {count} packages in {directory}.")


# ──────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def train(ctx):
    """YOLO11m training."""


@train.command("build")
@click.pass_context
def train_build(ctx):
    """Build the training dataset from all approved data."""
    from trainer.dataset import DatasetBuilder

    click.echo("Building training dataset...")
    builder = DatasetBuilder(ctx.obj["config"])

    stats = builder.get_stats()
    click.echo(f"  Total approved: {stats['total_approved']}")
    for cls, count in stats.get("by_class", {}).items():
        click.echo(f"    {cls:12s}: {count}")

    dataset_path = builder.build()
    if dataset_path:
        click.echo(f"\nDataset ready at: {dataset_path}")
    else:
        click.echo(
            "\nInsufficient data — keep collecting and reviewing.",
            err=True,
        )
        sys.exit(1)


@train.command("run")
@click.option("--dataset", default=None, help="Path to dataset directory.")
@click.pass_context
def train_run(ctx, dataset):
    """Build dataset (if needed) and run YOLO11m training."""
    from trainer.dataset import DatasetBuilder
    from trainer.train import YOLOTrainer

    # Build dataset if not specified
    if dataset is None:
        click.echo("Building training dataset...")
        builder = DatasetBuilder(ctx.obj["config"])
        dataset_path = builder.build()
        if not dataset_path:
            click.echo("Not enough approved data to train.", err=True)
            sys.exit(1)
    else:
        dataset_path = Path(dataset)

    click.echo(f"Starting YOLO11m training on {dataset_path}...")
    trainer = YOLOTrainer(ctx.obj["config"])
    result = trainer.train(dataset_path)

    click.echo(f"\nTraining complete — version: {result['version']}")
    click.echo(f"  mAP@50:     {result['metrics'].get('map50', 0):.4f}")
    click.echo(f"  mAP@50-95:  {result['metrics'].get('map50_95', 0):.4f}")

    if result["passed_quality_gates"]:
        click.secho("  Quality gates: PASSED ✓", fg="green")
        click.echo(f"\nDeploy with:\n  python cli.py deploy install {result['model_path']}")
    else:
        click.secho("  Quality gates: FAILED ✗", fg="red")
        click.echo("  Check metrics above. Not recommended for deployment.")


@train.command("evaluate")
@click.argument("model_path")
@click.option("--dataset", default="data/training_set", help="Dataset directory.")
@click.pass_context
def train_evaluate(ctx, model_path, dataset):
    """Evaluate a model against the training dataset."""
    from trainer.evaluate import ModelEvaluator

    evaluator = ModelEvaluator(ctx.obj["config"])

    if not Path(dataset).exists():
        click.echo(
            f"Dataset not found at {dataset}. Run 'python cli.py train build' first.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Evaluating {model_path}...")
    metrics = evaluator.evaluate(model_path, dataset)
    evaluator.print_report(metrics, version=Path(model_path).parent.name)


# ──────────────────────────────────────────────────────────
# DEPLOY
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def deploy(ctx):
    """Model export and deployment to Frigate."""


@deploy.command("export")
@click.argument("model_path")
@click.pass_context
def deploy_export(ctx, model_path):
    """Export a .pt model to ONNX format only (don't deploy to Frigate)."""
    from deployer.export import ModelExporter

    exporter = ModelExporter(ctx.obj["config"])
    model_dir = Path(model_path).parent

    click.echo(f"Exporting {model_path} to ONNX...")
    onnx_path = exporter.export(model_path)

    labelmap_path = model_dir / "labelmap.txt"
    exporter.write_labelmap(labelmap_path)

    click.echo(f"\nExport complete:")
    click.echo(f"  ONNX model: {onnx_path}")
    click.echo(f"  Labelmap:   {labelmap_path}")


@deploy.command("install")
@click.argument("model_path")
@click.pass_context
def deploy_install(ctx, model_path):
    """Export and deploy a model to Frigate config directory."""
    from deployer.deploy import ModelDeployer

    deployer = ModelDeployer(ctx.obj["config"])
    frigate_model_dir = deployer.frigate_model_dir

    click.echo(f"Deploying {model_path} to {frigate_model_dir}...")
    click.confirm(
        "This will replace the currently deployed model. Continue?",
        abort=True,
    )

    onnx_path = deployer.deploy(model_path)
    click.secho(f"\nModel deployed to {onnx_path}", fg="green")


@deploy.command("rollback")
@click.pass_context
def deploy_rollback(ctx):
    """Roll back to the previously deployed model."""
    from deployer.deploy import ModelDeployer

    deployer = ModelDeployer(ctx.obj["config"])
    click.confirm("Roll back to the previous model?", abort=True)

    if deployer.rollback():
        click.secho("Rollback complete. Restart Frigate to load the previous model.", fg="green")
    else:
        click.echo("Rollback failed — no archived model available.", err=True)
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def pipeline(ctx):
    """Overall pipeline status and orchestration."""


@pipeline.command("status")
@click.pass_context
def pipeline_status(ctx):
    """Show the complete pipeline dashboard."""
    from collector.db import Batch, Classification, Event, TrainingRun, get_session
    from sqlalchemy import func

    session = get_session()
    try:
        click.echo(f"\n{'═' * 60}")
        click.echo("  Frigate Training Pipeline — Dashboard")
        click.echo(f"{'═' * 60}")

        # Collection
        total_events = session.query(Event).count()
        click.echo(f"\n  COLLECTION")
        click.echo(f"    Total events:         {total_events}")

        # Classification
        total_classified = session.query(Classification).count()
        auto_approved = (
            session.query(Classification)
            .filter(Classification.decision.in_(["auto_approved", "auto_corrected"]))
            .count()
        )
        flagged = (
            session.query(Classification)
            .filter_by(decision="flagged_review", human_reviewed=False)
            .count()
        )
        human_reviewed = session.query(Classification).filter_by(human_reviewed=True).count()
        total_approved = session.query(Classification).filter_by(approved=True).count()

        click.echo(f"\n  CLASSIFICATION")
        click.echo(f"    Auto-approved:        {auto_approved}")
        click.echo(f"    Pending human review: {flagged}")
        click.echo(f"    Human reviewed:       {human_reviewed}")
        click.echo(f"    Total approved:       {total_approved}")

        # Per-class
        class_counts = (
            session.query(Classification.final_label, func.count(Classification.id))
            .filter_by(approved=True)
            .group_by(Classification.final_label)
            .all()
        )
        if class_counts:
            click.echo(f"\n  APPROVED PER CLASS")
            for label, count in sorted(class_counts):
                click.echo(f"    {label:20s}: {count}")

        # Training
        runs = (
            session.query(TrainingRun)
            .order_by(TrainingRun.id.desc())
            .limit(3)
            .all()
        )
        click.echo(f"\n  TRAINING RUNS (last 3)")
        if runs:
            for run in runs:
                status = "deployed" if run.deployed else ("PASS" if run.passed_quality_gates else "FAIL")
                map50 = f"{run.map50:.4f}" if run.map50 else "N/A"
                click.echo(f"    {run.version}  mAP50={map50}  [{status}]")
        else:
            click.echo("    No training runs yet.")

        # Readiness check
        click.echo(f"\n  READINESS")
        cfg_path = ctx.obj["config"]
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        min_images = cfg["training"]["min_training_images"]
        min_per_class = cfg["training"]["min_per_class"]

        ready = total_approved >= min_images
        click.echo(
            f"    Approved images: {total_approved}/{min_images} "
            f"({'READY' if ready else 'collecting...'})"
        )
        for label, count in sorted(class_counts):
            ok = count >= min_per_class
            click.echo(
                f"    {label:20s}: {count}/{min_per_class} "
                f"{'✓' if ok else '✗'}"
            )

        click.echo(f"{'═' * 60}\n")
    finally:
        session.close()


if __name__ == "__main__":
    cli(obj={})
