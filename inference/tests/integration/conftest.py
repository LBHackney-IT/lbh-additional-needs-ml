"""Fixtures for the pipeline integration tests.

`run_pipeline` runs the real gateway, preprocessor, span and relation pipelines,
and linker. Only boto3 and the model weights are replaced.
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import src.gateway.athena_gateway as athena_gateway
from src.common.schemas import EnrichedNote
from src.run_inference import InferenceConfig, run_inference
from src.services.relation_classifier_pipeline import RelationClassifierPipeline
from src.services.span_classifier_pipeline import SpanClassifierPipeline

LABEL_LIST = [
    "housing_conditions_hoarding",
    "mobility_mobility_physical",
    "physical_health",
    "property_level_disrepair_damp_mould",
    "person_name",
    "person_role",
]
SpanScript = tuple[str, str, float]
RelationScript = tuple[str, str]
COLUMNS = [
    "note_id",
    "note_category",
    "note_content",
    "note_date",
    "tenure_id",
    "household_members",
]


# --- Athena ---
class FakeAthenaClient:
    """Serve one result page built from row dicts, and record the query."""

    def __init__(self, rows: list[dict]):
        self.page = {
            "ResultSet": {
                "ResultSetMetadata": {"ColumnInfo": [{"Name": name} for name in COLUMNS]},
                "Rows": [self._cells(COLUMNS), *(self._cells(self._values(r)) for r in rows)],
            }
        }
        self.query: str | None = None

    @staticmethod
    def _values(row: dict) -> list[str | None]:
        household = row.get("household_members")
        if household is not None and not isinstance(household, str):
            household = json.dumps(household)
        return [*(row.get(name) for name in COLUMNS[:-1]), household]

    @staticmethod
    def _cells(values) -> dict:
        return {"Data": [{"VarCharValue": v} if v is not None else {} for v in values]}

    def start_query_execution(self, **kwargs):
        self.query = kwargs["QueryString"]
        return {"QueryExecutionId": "query-1"}

    def get_query_execution(self, **_kwargs):
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_paginator(self, name):
        assert name == "get_query_results"
        return type("Paginator", (), {"paginate": lambda *_a, **_k: [self.page]})()


# --- Dummy model weights ---
class WordOffsetTokenizer:
    """Tokenize on whitespace and remember the character offsets for the model."""

    def __init__(self):
        self.texts: list[str] = []
        self.offsets: list[list[tuple[int, int]]] = []

    def __call__(self, texts, **_kwargs):
        rows = []
        for text in texts:
            words = []
            for match in re.finditer(r"\S+", text):
                start, end = match.span()
                while end > start and text[end - 1] in ".,!?;:":
                    end -= 1
                if end > start:
                    words.append((start, end))
            rows.append([(0, 0), *words, (0, 0)])
        width = max(len(row) for row in rows)
        self.texts = list(texts)
        self.offsets = [row + [(0, 0)] * (width - len(row)) for row in rows]
        count = len(texts)
        return {
            "input_ids": torch.ones(count, width, dtype=torch.long),
            "attention_mask": torch.ones(count, width, dtype=torch.long),
            "offset_mapping": torch.tensor(self.offsets, dtype=torch.long),
            "overflow_to_sample_mapping": torch.arange(count),
        }


class ScriptedSpanModel:
    """Return configured labels and confidences for exact candidate text."""

    def __init__(self, tokenizer: WordOffsetTokenizer, scripts: list[SpanScript]):
        self.tokenizer = tokenizer
        self.scripts = {text: (label, confidence) for text, label, confidence in scripts}

    def __call__(self, _input_ids, _attention_mask, candidate_spans):
        background = len(LABEL_LIST)
        class_count = background + 1
        logits = []
        for chunk_idx, token_start, token_end in candidate_spans.tolist():
            offsets = self.tokenizer.offsets[chunk_idx]
            char_start, char_end = offsets[token_start][0], offsets[token_end - 1][1]
            snippet = self.tokenizer.texts[chunk_idx][char_start:char_end]
            scores = [0.0] * (background + 1)
            script = self.scripts.get(snippet)
            if script:
                label, confidence = script
                confidence = min(max(confidence, 1e-6), 1 - 1e-6)
                scores[LABEL_LIST.index(label)] = math.log(
                    confidence * (class_count - 1) / (1 - confidence)
                )
            else:
                scores[background] = 8.0
            logits.append(scores)
        return {"logits": torch.tensor(logits, dtype=torch.float)}


class PairTokenizer:
    def __init__(self, positive_pairs: set[RelationScript]):
        self.positive_pairs = positive_pairs

    def __call__(self, texts, **_kwargs):
        count = len(texts)
        input_ids = torch.zeros(count, 4, dtype=torch.long)
        for index, text in enumerate(texts):
            need = re.search(r"\[N_START\](.*?)\[N_END\]", text)
            person = re.search(r"\[P_START\](.*?)\[P_END\]", text)
            pair = (need.group(1), person.group(1)) if need and person else None
            input_ids[index, 0] = int(pair in self.positive_pairs)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(count, 4, dtype=torch.long),
            "overflow_to_sample_mapping": torch.arange(count),
        }


class PositiveRelationModel:
    def __call__(self, input_ids=None, attention_mask=None):
        positive = input_ids[:, 0].bool()
        logits = torch.zeros(input_ids.shape[0], 2)
        logits[:, 0] = torch.where(positive, 0.0, 5.0)
        logits[:, 1] = torch.where(positive, 5.0, 0.0)
        return type("Output", (), {"logits": logits})()


def _inject_dummy_models(
    monkeypatch,
    spans: list[SpanScript],
    relations: set[RelationScript],
) -> None:
    """Pass dummy weights into the real pipeline constructors."""
    span_tokenizer = WordOffsetTokenizer()
    span_model = ScriptedSpanModel(span_tokenizer, spans)
    span_init = SpanClassifierPipeline.__init__
    relation_init = RelationClassifierPipeline.__init__

    def init_span(self, model_dir=None, logger=None, **_kwargs):
        span_init(
            self,
            model_dir=model_dir,
            logger=logger,
            tokenizer=span_tokenizer,
            model=span_model,
            run_config={"base_model": "fake-bert", "max_length": 128, "max_candidate_size": 4},
            label_list=LABEL_LIST,
            thresholds=dict.fromkeys(LABEL_LIST, 0.5),
        )

    def init_relation(self, model_dir=None, threshold=0.5, **_kwargs):
        relation_init(
            self,
            model_dir=model_dir,
            max_length=32,
            threshold=threshold,
            tokenizer=PairTokenizer(relations),
            model=PositiveRelationModel(),
        )

    monkeypatch.setattr(SpanClassifierPipeline, "__init__", init_span)
    monkeypatch.setattr(RelationClassifierPipeline, "__init__", init_relation)


# --- Pipeline wrapper ---
@dataclass
class PipelineRun:
    config: InferenceConfig
    query: str
    raw_parts: list[list[dict]]
    parts: list[list[EnrichedNote]]

    @property
    def notes(self) -> list[EnrichedNote]:
        return [note for part in self.parts for note in part]

    @property
    def note(self) -> EnrichedNote:
        (only,) = self.notes
        return only

    @property
    def raw_note(self) -> dict:
        (only,) = [note for part in self.raw_parts for note in part]
        return only


def _read_parts(output_dir: Path) -> tuple[list[list[dict]], list[list[EnrichedNote]]]:
    """Read every flushed part and validate each note against the output schema."""
    raw_parts = [
        json.loads(path.read_text("utf-8")) for path in sorted(output_dir.glob("part_*.json"))
    ]
    validated = [[EnrichedNote.model_validate(note) for note in part] for part in raw_parts]
    return raw_parts, validated


@pytest.fixture
def run_pipeline(monkeypatch, tmp_path):
    """Return a function that runs the pipeline over Athena rows and reads the output."""

    def run(
        rows: list[dict],
        spans: list[SpanScript] | None = None,
        *,
        relations: set[RelationScript] | None = None,
        batch_size: int = 8,
        flush_every_n_batches: int = 50,
    ) -> PipelineRun:
        client = FakeAthenaClient(rows)
        monkeypatch.setattr(athena_gateway.boto3, "client", lambda *_a, **_k: client)
        _inject_dummy_models(monkeypatch, spans or [], relations or set())

        config = InferenceConfig(
            output_dir=tmp_path / "output",
            model_root=tmp_path / "models",
            athena_table="additional_needs_notes_reshaped",
            athena_where="note_date >= DATE '2024-01-01'",
            batch_size=batch_size,
            flush_every_n_batches=flush_every_n_batches,
        )
        run_inference(config, logging.getLogger("test"))
        raw_parts, validated = _read_parts(config.output_dir)
        return PipelineRun(config, client.query, raw_parts, validated)

    return run
