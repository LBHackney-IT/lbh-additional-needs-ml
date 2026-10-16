import json
import logging
from datetime import UTC, datetime

import pytest

from src.common.schemas import Link, Relation, Span, SpanPredictions
from src.gateway.athena_gateway import DEFAULT_WHERE
from src.run_inference import (
    InferenceConfig,
    _process_batch,
    parse_args,
    run_inference,
)


# --- Fixtures ---
class FakeSpanPipeline:
    def __init__(self, **_kwargs):
        pass

    def predict_batch(self, texts):
        return [
            SpanPredictions(
                needs=[
                    Span(
                        id=f"need-{index}",
                        text="mobility needs",
                        start=0,
                        end=14,
                        label="physical_health",
                        confidence=0.9,
                    )
                ],
                persons=[
                    Span(
                        id=f"person-span-{index}",
                        text="Jane Doe",
                        start=15,
                        end=23,
                        label="person_name",
                        confidence=0.95,
                    )
                ],
            )
            for index, _text in enumerate(texts)
        ]


class FakeRelationPipeline:
    def __init__(self, **_kwargs):
        self.calls = []

    def predict_from_spans(self, text, needs, persons):
        self.calls.append((text, needs, persons))
        return [
            Relation(
                from_id=needs[0].id,
                to=persons[0].id,
                confidence=0.85,
            )
        ]


class FakeLinker:
    def __init__(self, **_kwargs):
        pass

    def link(self, note):
        return [
            Link(
                need_id=note.needs[0].id,
                target_id="person-1",
                target_type="person",
                confidence=0.85,
                method="test_link",
            )
        ]


class FakeAthenaSource:
    records = [
        {
            "note_id": "note-1",
            "note_content": "first",
            "note_date": None,
            "tenure_id": None,
            "household_members": [],
        },
        {
            "note_id": "note-2",
            "note_content": "second",
            "note_date": None,
            "tenure_id": None,
            "household_members": [],
        },
    ]
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.instances.append(self)

    def fetch_records(self):
        return self.records


# --- Parsing args ---
def test_parse_args_uses_env_athena_table(monkeypatch):
    monkeypatch.setenv("ATHENA_TABLE", "environment_table")
    config = parse_args([])

    assert config.athena_table == "environment_table"


def test_parse_args_uses_cli_athena_table(monkeypatch):
    monkeypatch.setenv("ATHENA_TABLE", "environment_table")
    config = parse_args(["--athena-table", "v_repairs_jds_reshaped"])

    assert config.athena_table == "v_repairs_jds_reshaped"


def test_parse_args_defaults_to_the_mmh_where_clause(monkeypatch):
    monkeypatch.delenv("ATHENA_WHERE", raising=False)
    config = parse_args([])

    assert config.athena_where == DEFAULT_WHERE


def test_parse_args_uses_cli_where_clause(monkeypatch):
    monkeypatch.setenv("ATHENA_WHERE", "flag_inactive = FALSE")
    config = parse_args(["--where", "note_date >= DATE '2024-01-01'"])

    assert config.athena_where == "note_date >= DATE '2024-01-01'"


def test_parse_args_uses_env_where_clause(monkeypatch):
    monkeypatch.setenv("ATHENA_WHERE", "note_date < DATE '2025-01-01'")
    config = parse_args([])

    assert config.athena_where == "note_date < DATE '2025-01-01'"


def test_parse_args_defaults_source_to_mmh(monkeypatch):
    monkeypatch.delenv("SOURCE", raising=False)
    config = parse_args([])

    assert config.source == "mmh"


def test_parse_args_uses_env_source(monkeypatch):
    monkeypatch.setenv("SOURCE", "mmh")

    config = parse_args(["--source", "repairs_hub"])

    assert config.source == "repairs_hub"


def test_parse_args_rejects_an_unsafe_source():
    with pytest.raises(ValueError, match="Invalid source"):
        parse_args(["--source", "repairs hub"])


# --- Pipeline Unit Tests ----
def test_process_batch_calls_each_stage():
    relation_pipeline = FakeRelationPipeline()
    record = {
        "id": "note-1",
        "text": "Jane Doe mobility needs",
        "date": "2026-01-01",
        "tenure_ids": ["tenure-1"],
        "household_members": [],
    }

    payloads = _process_batch(
        [record],
        FakeSpanPipeline(),
        relation_pipeline,
        FakeLinker(),
        "span_relation",
        datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert relation_pipeline.calls[0][1][0].label == "physical_health"
    assert relation_pipeline.calls[0][2][0].label == "person_name"
    assert payloads[0]["relations"][0]["from"] == "need-0"
    assert payloads[0]["links"][0]["target_id"] == "person-1"


def test_run_inference_flushes_each_batch(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"

    import src.run_inference as run_module

    FakeAthenaSource.instances = []
    monkeypatch.setattr(run_module, "AthenaRecordSource", FakeAthenaSource)
    monkeypatch.setattr(run_module, "SpanClassifierPipeline", FakeSpanPipeline)
    monkeypatch.setattr(run_module, "RelationClassifierPipeline", FakeRelationPipeline)
    monkeypatch.setattr(run_module, "DeterministicEntityLinker", FakeLinker)

    run_inference(
        InferenceConfig(
            output_dir=output_dir,
            model_root=tmp_path / "models",
            athena_database="housing-refined-zone",
            athena_table="v_repairs_jds_reshaped",
            athena_workgroup="housing",
            aws_region="eu-west-2",
            batch_size=1,
            flush_every_n_batches=1,
        ),
        logging.getLogger("test"),
    )

    first_part = json.loads((output_dir / "part_0001.json").read_text(encoding="utf-8"))
    second_part = json.loads((output_dir / "part_0002.json").read_text(encoding="utf-8"))
    assert FakeAthenaSource.instances[0].config["table"] == "v_repairs_jds_reshaped"
    assert [first_part[0]["id"], second_part[0]["id"]] == ["note-1", "note-2"]
