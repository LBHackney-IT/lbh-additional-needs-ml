"""Production batch inference entrypoint for SageMaker Processing / Airflow.

Executes the batch inference pipeline:
  1. Queries Athena
  2. Preprocessing
  3. Span extraction (NER) via DeBERTa
  4. Relation extraction via ALBERT
  5. Deterministic entity linking
  6. Postprocessing & output as JSON

Flushes partitioned JSON output every N batches to limit RAM usage. Notes are
processed in note_date order, and each flush logs the last date in that part.
After a failed run, resume by adding that date to the Athena WHERE clause.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.common.logging import setup_logger
from src.common.schemas import DomainNote
from src.gateway.athena_gateway import DEFAULT_WHERE, AthenaRecordSource
from src.services.entity_linker import DeterministicEntityLinker
from src.services.note_preprocessor import NotePreprocessor
from src.services.relation_classifier_pipeline import RelationClassifierPipeline
from src.services.span_classifier_pipeline import SpanClassifierPipeline

# SageMaker Processing default channel paths.
DEFAULT_MODEL_ROOT = Path("/opt/ml/processing/model")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/processing/output")


@dataclass
class InferenceConfig:
    output_dir: Path
    model_root: Path
    athena_database: str = "housing-refined-zone"
    athena_table: str = "additional_needs_notes_reshaped"
    athena_workgroup: str = "housing"
    athena_where: str = DEFAULT_WHERE
    source: str = "mmh"
    aws_region: str = "eu-west-2"
    batch_size: int = 128
    flush_every_n_batches: int = 50
    relation_threshold: float = 0.5
    span_model_name: str = "span_v1"
    relation_model_name: str = "relation_v1"


# === Helpers ====
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


def _flush_part(
    output_dir: Path,
    part_idx: int,
    payloads: list[dict],
    total_words: int,
    total_pairs_scored: int,
    logger,
) -> None:
    """Flush a batch partition of enriched note records to disk as formatted JSON.

    The part is written to a sibling ``.tmp`` file and replaces the final name
    only after the write is flushed. A failure mid-write leaves the previous
    part file untouched.

    Args:
        output_dir: Target output directory.
        part_idx: Partition index number.
        payloads: List of serialized enriched note dictionaries.
        total_words: Whitespace word count of the notes in this part.
        total_pairs_scored: Need-person pairs sent to the relation model.
        logger: Operational logger.
    """
    if not payloads:
        return

    json_path = output_dir / f"part_{part_idx:04d}.json"
    temporary_path = json_path.with_name(f".{json_path.name}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, json_path)

    last_date = payloads[-1].get("date") or "none"
    logger.info(
        "Flushed part %04d: %d notes, %d words, %d candidate pairs through note_date %s",
        part_idx,
        len(payloads),
        total_words,
        total_pairs_scored,
        last_date,
    )


# =========================
# Main Inference Logic
# =========================


def _process_batch(
    records: list[DomainNote],
    span_pipeline: SpanClassifierPipeline,
    relation_pipeline: RelationClassifierPipeline,
    linker: DeterministicEntityLinker,
    model_name: str,
    pipeline_run_at: datetime,
) -> tuple[list[dict], int, int]:
    """Execute stages 3-6 over a micro-batch of domain notes.

    Args:
        records: Batch of domain notes.
        span_pipeline: Initialized span extraction pipeline.
        relation_pipeline: Initialized relation classification pipeline.
        linker: Initialized deterministic entity linker engine.
        model_name: Composite identifier string for model versions.
        pipeline_run_at: Timestamp representing pipeline execution start.

    Returns:
        Serialized enriched notes, their word count, and the need-person pairs scored.
    """
    texts = [record.expanded_text for record in records]
    enriched_payloads: list[dict] = []
    word_count = 0
    pairs_scored = 0

    # 1. Span Classifier
    span_preds = span_pipeline.predict_batch(texts)

    for record, text, spans in zip(records, texts, span_preds, strict=True):
        needs = spans.needs
        persons = spans.persons
        word_count += len(text.split())
        pairs_scored += len(relation_pipeline._build_candidate_pairs(text, needs, persons))
        # 2. Relation Classifier
        relations = relation_pipeline.predict_from_spans(text, needs, persons)
        record.needs = needs
        record.persons = persons
        record.relations = relations
        # 3. Entity Linking
        record.links = linker.link(record)
        # 4. Convert to output
        note = record.to_enriched(model=model_name, pipeline_run_at=pipeline_run_at)
        enriched_payloads.append(note.model_dump_for_json())

    return enriched_payloads, word_count, pairs_scored


def run_inference(config: InferenceConfig, logger) -> None:
    """Orchestrate batch inference over dataset inputs.

    Args:
        config: InferenceConfig instance holding paths and runtime hyperparameters.
        logger: Logger instance for reporting.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Query production data through Athena
    record_source = AthenaRecordSource(
        database=config.athena_database,
        table=config.athena_table,
        workgroup=config.athena_workgroup,
        region=config.aws_region,
        where=config.athena_where,
        logger=logger,
    )
    rows = record_source.fetch_records()

    # 2. Preprocess, then drop the raw rows. The notes list holds what inference needs.
    records = NotePreprocessor(source=config.source).preprocess_notes(rows)
    logger.info("Loaded %d note-tenure rows from Athena (%d notes)", len(rows), len(records))
    del rows  # Drop original data copies - memory saving

    # 3. Load models
    span_model_dir = config.model_root / config.span_model_name / "final_model"
    relation_model_dir = config.model_root / config.relation_model_name / "final_model"

    span_pipeline = SpanClassifierPipeline(model_dir=span_model_dir, logger=logger)
    relation_pipeline = RelationClassifierPipeline(
        model_dir=relation_model_dir,
        threshold=config.relation_threshold,
    )
    linker = DeterministicEntityLinker(logger=logger)

    model_name = f"{config.span_model_name}_{config.relation_model_name}"
    pipeline_run_at = datetime.now(UTC)
    logger.info("Loaded models. Model name: %s; Run datetime: %s", model_name, pipeline_run_at)

    # 4. Start micro-batching notes
    dataset = NotesDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate_records,
        num_workers=0,
    )

    buffered_payloads: list[dict] = []
    part_words = 0
    part_pairs = 0
    part_idx = 1
    batch_counter = 0

    with torch.no_grad():
        for batch in loader:
            enriched, word_count, pairs_scored = _process_batch(
                batch,
                span_pipeline,
                relation_pipeline,
                linker,
                model_name,
                pipeline_run_at,
            )
            buffered_payloads.extend(enriched)
            part_words += word_count
            part_pairs += pairs_scored
            batch_counter += 1

            # 5. Flush periodically (and log)
            if batch_counter % config.flush_every_n_batches == 0:
                _flush_part(output_dir, part_idx, buffered_payloads, part_words, part_pairs, logger)
                buffered_payloads = []
                part_words = 0
                part_pairs = 0
                part_idx += 1

    # 6. Final flush
    if buffered_payloads:
        _flush_part(output_dir, part_idx, buffered_payloads, part_words, part_pairs, logger)

    logger.info("Inference complete. Output written to %s", output_dir)


# ===================
# Entrypoint
# ===================


def _validate_source(source: str) -> str:
    """Return a source tag that is safe to stamp onto output records."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source):
        raise ValueError(f"Invalid source: {source!r}")
    return source


def _validate_model_name(name: str, label: str) -> str:
    """Return a model directory name that cannot escape the model root."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


def parse_args(argv: list[str] | None = None) -> InferenceConfig:
    """Parse command line arguments and environment variables into InferenceConfig.

    Args:
        argv: Optional list of command-line argument strings.

    Returns:
        Populated InferenceConfig object.
    """
    # Athena Args
    parser = argparse.ArgumentParser(description="Additional Needs batch inference pipeline")
    parser.add_argument(
        "--athena-database",
        default=os.environ.get("ATHENA_DATABASE", "prod__housing-refined-zone"),
        help="Athena database containing the production note table",
    )
    parser.add_argument(
        "--athena-table",
        default=os.environ.get("ATHENA_TABLE", "additional_needs_notes_reshaped"),
        help="Athena table containing shaped note records",
    )
    parser.add_argument(
        "--athena-workgroup",
        default=os.environ.get("ATHENA_WORKGROUP", "housing"),
        help="Athena workgroup used for production queries",
    )
    parser.add_argument(
        "--where",
        default=os.environ.get("ATHENA_WHERE", DEFAULT_WHERE),
        help=(
            "SQL predicate applied as the Athena WHERE clause. "
            "Defaults to the MMH flag filters. Pass an empty string to read every row. "
            "ATHENA_WHERE is the environment equivalent."
        ),
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("SOURCE", "mmh"),
        help="Source system stamped on every output note. Defaults to mmh.",
    )
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION", "eu-west-2"),
        help="AWS region for the Athena client",
    )
    # Model Args
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
        help="Root directory containing one directory per model name",
    )
    parser.add_argument(
        "--span-model-name",
        default=os.environ.get("SPAN_MODEL_NAME", "span_v1"),
        help="Directory under the model root containing the span model. Defaults to span_v1.",
    )
    parser.add_argument(
        "--relation-model-name",
        default=os.environ.get("RELATION_MODEL_NAME", "relation_v1"),
        help=(
            "Directory under the model root containing the relation model. Defaults to relation_v1."
        ),
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
        output_dir=args.output_dir,
        model_root=args.model_root,
        athena_database=args.athena_database,
        athena_table=args.athena_table,
        athena_workgroup=args.athena_workgroup,
        athena_where=args.where,
        source=_validate_source(args.source),
        aws_region=args.aws_region,
        batch_size=args.batch_size,
        flush_every_n_batches=args.flush_every_n_batches,
        relation_threshold=args.relation_threshold,
        span_model_name=_validate_model_name(args.span_model_name, "span model name"),
        relation_model_name=_validate_model_name(args.relation_model_name, "relation model name"),
    )


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for running batch inference from CLI or orchestrator.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Exit code (0 for success).
    """
    config = parse_args(argv)
    logger = setup_logger("inference.pipeline", "pipeline.log", config.output_dir / "logs")
    logger.info(
        "Starting inference: source=%s athena=%s.%s output=%s model_root=%s",
        config.source,
        config.athena_database,
        config.athena_table,
        config.output_dir,
        config.model_root,
    )
    run_inference(config, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
