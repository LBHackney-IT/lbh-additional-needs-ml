import json

import numpy as np
import torch

from src.common.schemas import Span, SpanPredictions
from src.services.span_classifier_pipeline import SpanClassifierPipeline, _cast_for_device


# -- Helpers --
class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        assert texts == ["one two"]
        return {
            "input_ids": torch.tensor([[101, 1, 2, 102]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "offset_mapping": torch.tensor([[[0, 0], [0, 3], [4, 7], [0, 0]]]),
            "overflow_to_sample_mapping": torch.tensor([0]),
        }


class FakeSpanModel:
    def __call__(self, _input_ids, _attention_mask, candidate_spans):
        assert candidate_spans.tolist() == [[0, 1, 2], [0, 2, 3]]
        return {"logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]])}


def make_span_pipeline():
    return SpanClassifierPipeline(
        model_dir="mock/path",
        stride=128,
        run_config={"base_model": "fake-bert", "max_length": 128, "max_candidate_size": 1},
        label_list=["physical_health"],
        thresholds={"physical_health": 0.5},
        tokenizer=FakeTokenizer(),
        model=FakeSpanModel(),
    )


# --- Candidate generation ---
def test_generate_candidates_returns_contiguous_windows():
    # Token 0 & 4 are special [0-width]. Real tokens are 1, 2, 3.
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)]

    candidates = SpanClassifierPipeline._generate_candidates(offsets, max_size=2)

    assert candidates == [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)]


def test_generate_candidates_skips_special_tokens_and_token_gaps():
    # Special token at index 3 breaks contiguity; (2, 4) window is rejected.
    offsets = [(0, 0), (0, 2), (2, 4), (0, 0), (8, 10)]

    candidates = SpanClassifierPipeline._generate_candidates(offsets, max_size=2)

    assert candidates == [(1, 2), (2, 3), (4, 5), (1, 3)]


def test_generate_candidates_returns_no_windows_for_empty_or_zero_size_input():
    empty_offsets = SpanClassifierPipeline._generate_candidates([], max_size=2)
    zero_size = SpanClassifierPipeline._generate_candidates([(0, 2)], max_size=0)

    assert empty_offsets == []
    assert zero_size == []


# --- Dtype ---
def test_cuda_cast_makes_a_float_head_match_the_half_backbone():
    model = torch.nn.Linear(4, 2)
    model.half()
    head = torch.nn.Linear(2, 1)
    combined = torch.nn.Sequential(model, head)

    _cast_for_device(combined, torch.device("cuda"))

    assert combined[0].weight.dtype == torch.float16
    assert combined[1].weight.dtype == torch.float16


def test_cpu_cast_upcasts_a_float16_checkpoint():
    model = torch.nn.Linear(4, 2).half()

    _cast_for_device(model, torch.device("cpu"))

    assert model.weight.dtype == torch.float32


# --- Load thresholds ---
def test_load_thresholds_reads_saved_class_thresholds(tmp_path):
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps({"physical_health": 0.73}), encoding="utf-8")

    thresholds = SpanClassifierPipeline._load_thresholds(
        thresholds_path, ["physical_health"], logger=None
    )

    assert thresholds == {"physical_health": 0.73}


def test_load_thresholds_defaults_each_threshold_to_fifty_when_file_is_missing(tmp_path):
    missing_path = tmp_path / "missing-thresholds.json"

    thresholds = SpanClassifierPipeline._load_thresholds(
        missing_path, ["physical_health", "person_name"], logger=None
    )

    assert thresholds == {"physical_health": 0.5, "person_name": 0.5}


# --- Prediction ---
def test_predict_batch_runs_predict_and_formats_correctly():
    results = make_span_pipeline().predict_batch(["one two"])

    assert isinstance(results[0], SpanPredictions)
    assert len(results[0].needs) == 1

    prediction = results[0].needs[0]
    assert prediction.text == "one"
    assert prediction.start == 0
    assert prediction.end == 3
    assert prediction.label == "physical_health"
    assert prediction.confidence > 0.5
    assert results[0].persons == []


# --- Apply threshold and format ---
def test_threshold_and_format_drops_no_entity_and_low_confidence():
    predictions = SpanClassifierPipeline._threshold_and_format(
        "need person",
        [(0, 4), (5, 11), (0, 4)],
        np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6]]),
        ["physical_health"],
        {"physical_health": 0.8},
    )

    assert len(predictions) == 1
    assert predictions[0].text == "need"
    assert predictions[0].label == "physical_health"


def test_threshold_and_format_applies_the_threshold_for_each_predicted_label():
    probabilities = np.array(
        [
            [0.76, 0.10, 0.14],  # physical_health (max i=0) w/ prob 0.76 (>0.75 - accept)
            [0.10, 0.76, 0.14],  # person_name (max i=1) w/ prob 0.76 (<0.8 - reject)
            [
                0.10,
                0.10,
                0.80,
            ],  # no_entity (max i=2) w/ prob 0.8 (no output - we don't label no entity)
        ]
    )

    predictions = SpanClassifierPipeline._threshold_and_format(
        "need person",
        [(0, 4), (5, 11), (0, 4)],
        probabilities,
        ["physical_health", "person_name"],
        {"physical_health": 0.75, "person_name": 0.80},
    )

    assert len(predictions) == 1
    assert predictions[0].label == "physical_health"


def test_threshold_and_format_returns_empty_for_missing_probabilities_or_spans():
    probs = SpanClassifierPipeline._threshold_and_format(
        "text", [(0, 4)], None, ["physical_health"], {"physical_health": 0.5}
    )
    spans = SpanClassifierPipeline._threshold_and_format(
        "text", [], np.array([[0.9, 0.1]]), ["physical_health"], {"physical_health": 0.5}
    )

    assert probs == []
    assert spans == []


# --- Deduplication ---
def test_deduplicate_predictions_keeps_highest_confidence_overlapping_same_label():
    predictions = [
        Span(id="low", text="needs", start=0, end=5, label="need", confidence=0.7),
        Span(id="high", text="needs", start=0, end=5, label="need", confidence=0.9),
        Span(id="person", text="needs", start=0, end=5, label="person_name", confidence=0.8),
    ]

    deduplicated = SpanClassifierPipeline._deduplicate_predictions(predictions)
    assert [prediction.id for prediction in deduplicated] == ["high", "person"]


def test_deduplicate_predictions_keeps_adjacent_same_label_spans():
    predictions = [
        Span(id="first", text="one", start=0, end=3, label="need", confidence=0.8),
        Span(id="second", text="two", start=3, end=6, label="need", confidence=0.7),
    ]

    deduplicated = SpanClassifierPipeline._deduplicate_predictions(predictions)
    assert [prediction.id for prediction in deduplicated] == ["first", "second"]


def test_deduplicate_predictions_returns_empty_for_no_predictions():
    deduplicated = SpanClassifierPipeline._deduplicate_predictions([])
    assert deduplicated == []


# --- Whitespace offset fixing ---
def test_correct_offset_strips_leading_whitespace():
    assert SpanClassifierPipeline._correct_offset("  need", 0, 6) == (2, 6)


def test_correct_offset_leaves_non_whitespace_start_unchanged():
    assert SpanClassifierPipeline._correct_offset("need", 0, 4) == (0, 4)


# --- Edge cases ---
def test_predict_batch_returns_no_predictions_for_empty_input():
    pipeline = make_span_pipeline()
    results = pipeline.predict_batch([])

    assert results == []


def test_predict_batch_returns_empty_predictions_when_no_real_tokens_exist():
    class EmptyTokenizer:
        def __call__(self, texts, **_kwargs):
            return {
                "input_ids": torch.ones(len(texts), 1, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 1, dtype=torch.long),
                "offset_mapping": torch.zeros(len(texts), 1, 2, dtype=torch.long),
                "overflow_to_sample_mapping": torch.arange(len(texts)),
            }

    class UnexpectedModelCall:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("The model should not run without candidate spans")

    pipeline = SpanClassifierPipeline(
        model_dir="mock/path",
        run_config={"base_model": "fake-bert", "max_length": 128, "max_candidate_size": 1},
        label_list=["physical_health"],
        thresholds={"physical_health": 0.5},
        tokenizer=EmptyTokenizer(),
        model=UnexpectedModelCall(),
    )

    results = pipeline.predict_batch(["", ""])
    assert results == [SpanPredictions(), SpanPredictions()]
