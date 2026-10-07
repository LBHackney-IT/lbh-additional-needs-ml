"""Production batch inference entrypoint for SageMaker Processing / Airflow.

Executes a 3-stage ML inference pipeline:
  1. Span extraction (NER) via DeBERTa
  2. Relation extraction via ALBERT
  3. Deterministic entity linking via rule engine

Flushes partitioned JSON output every N batches to limit RAM usage and allow
the pipeline to resume on failure
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from models.entity_linker import DeterministicEntityLinker
from models.relation_classifier import RelationClassifierPipeline
from models.span_classifier import SpanClassifierPipeline
from torch.utils.data import DataLoader, Dataset

from src.common.json_helpers import load_json
from src.common.logging import setup_logger
from src.common.schemas import EnrichedNote

# SageMaker Processing default channel paths.
DEFAULT_MODEL_ROOT = Path("/opt/ml/processing/model")
DEFAULT_INPUT_DIR = Path("/opt/ml/processing/input")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/processing/output")


@dataclass
class InferenceConfig:
    input_dir: Path
    output_dir: Path
    model_root: Path
    batch_size: int = 128
    flush_every_n_batches: int = 50
    relation_threshold: float = 0.5
    span_model_name: str = "span"
    relation_model_name: str = "relation"


class NotesDataset(Dataset):
    """Dataset wrapper for streaming document records through a PyTorch DataLoader."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def _collate_records(batch: list[dict]) -> list[dict]:
    """Collate function to preserve dictionaries in DataLoader batches."""
    return batch


def _discover_input_records(input_dir: Path, logger) -> list[dict]:
    """Load and aggregate document JSON records from the input directory.

    Args:
        input_dir: Directory containing input JSON files.
        logger: Logger instance for reporting.

    Returns:
        Flattened list of document dictionary records.
    """
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON input files found in {input_dir}")

    records: list[dict] = []
    for path in json_files:
        payload = load_json(path, logger)
        if isinstance(payload, list):
            records.extend(payload)
        else:
            records.append(payload)
    return records


def _flush_part(output_dir: Path, part_idx: int, payloads: list[dict], logger) -> None:
    """Flush a batch partition of enriched note records to disk as formatted JSON.

    Args:
        output_dir: Target output directory.
        part_idx: Partition index number.
        payloads: List of serialized enriched note dictionaries.
        logger: Operational logger.
    """
    if not payloads:
        return

    json_path = output_dir / f"part_{part_idx:04d}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2, ensure_ascii=False)

    logger.info("Flushed part %04d: %d notes", part_idx, len(payloads))


def _process_batch(
    records: list[dict],
    span_pipeline: SpanClassifierPipeline,
    relation_pipeline: RelationClassifierPipeline,
    linker: DeterministicEntityLinker,
    model_name: str,
    pipeline_run_at: datetime,
) -> list[dict]:
    """Execute stages 1-3 over a micro-batch of note records.

    Args:
        records: Batch of raw note dictionaries.
        span_pipeline: Initialized span extraction pipeline.
        relation_pipeline: Initialized relation classification pipeline.
        linker: Initialized deterministic entity linker engine.
        model_name: Composite identifier string for model versions.
        pipeline_run_at: Timestamp representing pipeline execution start.

    Returns:
        List of serialized enriched note dictionaries matching output schema.
    """
    texts = [r.get("text", "") for r in records]
    span_preds = span_pipeline.predict_batch(texts)  # 1. Span Classifier

    enriched_payloads: list[dict] = []
    for record, text, spans in zip(records, texts, span_preds, strict=True):
        needs = spans.needs
        persons = spans.persons
        relations = relation_pipeline.predict_from_spans(
            text, needs, persons
        )  # 2. Relation Classifier

        note = EnrichedNote(
            id=record.get("id", ""),
            text=text,
            date=record.get("note_date") or record.get("date"),
            model=model_name,
            needs=needs,
            persons=persons,
            relations=relations,
            tenure_ids=record.get("tenure_ids", []),
            household_members=record.get("household_members", []),
            pipeline_run_at=pipeline_run_at,
        )

        note.links = linker.link(note)  # 3. Entity Linker
        enriched_payloads.append(note.model_dump_for_json())

    return enriched_payloads


def run_inference(config: InferenceConfig, logger) -> None:
    """Orchestrate batch inference over dataset inputs.

    Args:
        config: InferenceConfig instance holding paths and runtime hyperparameters.
        logger: Logger instance for reporting.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 1. Get records from S3
    records = _discover_input_records(config.input_dir, logger)
    logger.info("Loaded %d notes from %s", len(records), config.input_dir)

    # 2. Load models
    span_model_dir = config.model_root / config.span_model_name / "final_model"
    relation_model_dir = config.model_root / config.relation_model_name / "final_model"

    span_pipeline = SpanClassifierPipeline(
        model_dir=span_model_dir, device=gpu_device, logger=logger
    )
    relation_pipeline = RelationClassifierPipeline(
        model_dir=relation_model_dir,
        threshold=config.relation_threshold,
        device=gpu_device,
    )
    linker = DeterministicEntityLinker(logger=logger)

    model_name = f"{config.span_model_name}_{config.relation_model_name}"
    pipeline_run_at = datetime.now(UTC)

    # 3. Start micro-batching notes
    dataset = NotesDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate_records,
        num_workers=0,
    )

    buffered_payloads: list[dict] = []
    part_idx = 1
    batch_counter = 0

    with torch.no_grad():
        for batch in loader:
            enriched = _process_batch(
                batch,
                span_pipeline,
                relation_pipeline,
                linker,
                model_name,
                pipeline_run_at,
            )
            buffered_payloads.extend(enriched)
            batch_counter += 1

            # Flush periodically
            if batch_counter % config.flush_every_n_batches == 0:
                _flush_part(output_dir, part_idx, buffered_payloads, logger)
                buffered_payloads = []
                part_idx += 1

    # Final flush
    if buffered_payloads:
        _flush_part(output_dir, part_idx, buffered_payloads, logger)

    logger.info("Inference complete. Output written to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> InferenceConfig:
    """Parse command line arguments and environment variables into InferenceConfig.

    Args:
        argv: Optional list of command-line argument strings.

    Returns:
        Populated InferenceConfig object.
    """
    parser = argparse.ArgumentParser(description="Additional Needs batch inference pipeline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("INPUT_DIR", DEFAULT_INPUT_DIR)),
        help="Directory containing input note JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        help="Directory for partitioned JSON output",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("MODEL_ROOT", DEFAULT_MODEL_ROOT)),
        help="Root directory containing span/ and relation/ model artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    parser.add_argument(
        "--flush-every-n-batches",
        type=int,
        default=int(os.environ.get("FLUSH_EVERY_N_BATCHES", "50")),
    )
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=float(os.environ.get("RELATION_THRESHOLD", "0.5")),
    )
    args = parser.parse_args(argv)

    return InferenceConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_root=args.model_root,
        batch_size=args.batch_size,
        flush_every_n_batches=args.flush_every_n_batches,
        relation_threshold=args.relation_threshold,
    )


def main(argv: list[str] | None = None) -> int:
    """Main execution entrypoint for running batch inference from CLI or orchestrator.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Exit code (0 for success).
    """
    config = parse_args(argv)
    logger = setup_logger("inference.pipeline", "pipeline.log")
    logger.info(
        "Starting inference: input=%s output=%s model_root=%s",
        config.input_dir,
        config.output_dir,
        config.model_root,
    )
    run_inference(config, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
