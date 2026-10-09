import numpy as np
import torch

from src.classifiers.span_classifier import (
    SpanClassifier,
    SpanClassifierPipeline,
    correct_offset,
    deduplicate_predictions,
    generate_candidates,
    threshold_and_format,
)
from src.common.schemas import Span, SpanPredictions


class DummySpanBackbone(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids=None, attention_mask=None):
        batch, seq = input_ids.shape
        hidden = torch.randn(batch, seq, self.config.hidden_size)
        return type("Output", (), {"last_hidden_state": hidden})()


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


def build_span_classifier():
    pipeline = SpanClassifierPipeline.__new__(SpanClassifierPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.base_model = "dummy-model"
    pipeline.max_length = 32
    pipeline.max_candidate_size = 1
    pipeline.label_list = ["physical_health"]
    pipeline.thresholds = {"physical_health": 0.5}
    pipeline.person_labels = ["person_name", "person_role"]
    pipeline.tokenizer = FakeTokenizer()
    pipeline.model = FakeSpanModel()
    return pipeline


def test_span_classifier_forward_pass_with_dummy_backbone(monkeypatch):
    monkeypatch.setattr(
        "src.classifiers.span_classifier.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: DummySpanBackbone(hidden_size=16),
    )

    model = SpanClassifier("dummy-model", num_labels=3)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    candidate_spans = torch.tensor([[0, 1, 3]])

    with torch.no_grad():
        output = model(input_ids, attention_mask, candidate_spans)

    assert output["logits"].shape == (1, 4)


def test_generate_candidates_returns_contiguous_windows():
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)]

    candidates = generate_candidates(offsets, max_size=2)

    assert (1, 2) in candidates
    assert (1, 3) in candidates


def test_predict_batch_runs_real_span_postprocessing():
    results = build_span_classifier().predict_batch(["one two"])

    assert isinstance(results[0], SpanPredictions)
    assert len(results[0].needs) == 1
    prediction = results[0].needs[0]
    assert prediction.text == "one"
    assert prediction.start == 0
    assert prediction.end == 3
    assert prediction.label == "physical_health"
    assert prediction.confidence > 0.5
    assert results[0].persons == []


def test_threshold_and_format_drops_no_entity_and_low_confidence():
    predictions = threshold_and_format(
        "need person",
        [(0, 4), (5, 11), (0, 4)],
        np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6]]),
        ["physical_health"],
        {"physical_health": 0.8},
    )

    assert len(predictions) == 1
    assert predictions[0].text == "need"
    assert predictions[0].label == "physical_health"


def test_deduplicate_predictions_keeps_highest_confidence_same_label():
    predictions = [
        Span(id="low", text="needs", start=0, end=5, label="need", confidence=0.7),
        Span(id="high", text="needs", start=0, end=5, label="need", confidence=0.9),
        Span(id="person", text="needs", start=0, end=5, label="person_name", confidence=0.8),
    ]

    deduplicated = deduplicate_predictions(predictions)

    assert [prediction.id for prediction in deduplicated] == ["high", "person"]


def test_correct_offset_strips_leading_whitespace():
    assert correct_offset("  need", 0, 6) == (2, 6)
