import pytest
import torch

from src.common.schemas import Span
from src.services.relation_classifier_pipeline import RelationClassifierPipeline


# --- Helpers ----
def make_span(span_id, text, start, end, label):
    return Span(id=span_id, text=text, start=start, end=end, label=label, confidence=0.9)


def make_relation_pipeline(window_logits, threshold=0.5):
    class FakeBatch(dict):
        def __init__(self, window_count):
            super().__init__(
                input_ids=torch.ones(window_count, 4, dtype=torch.long),
                attention_mask=torch.ones(window_count, 4, dtype=torch.long),
                overflow_to_sample_mapping=torch.zeros(window_count, dtype=torch.long),
            )

        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            assert len(texts) == 1
            return FakeBatch(len(window_logits))

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            return type("Output", (), {"logits": torch.tensor(window_logits)})()

    return RelationClassifierPipeline(
        model_dir="mock/path",
        max_length=32,
        stride=8,
        threshold=threshold,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )


# -- Build candidate pairs --
def test_predict_pairs_returns_empty_for_no_candidates():
    pipeline = RelationClassifierPipeline.__new__(RelationClassifierPipeline)

    assert pipeline.predict_pairs([]) == []


def test_build_candidate_pairs_returns_cartesian_product():
    need_one = make_span("need-1", "support", 0, 7, "physical_health")
    need_two = make_span("need-2", "housing", 8, 15, "housing")
    person = make_span("person-1", "Jane", 12, 16, "person_name")

    pairs = RelationClassifierPipeline._build_candidate_pairs(
        "support housing for Jane", [need_one, need_two], [person]
    )

    assert [(need.id, person.id) for _, need, person in pairs] == [
        ("need-1", "person-1"),
        ("need-2", "person-1"),
    ]


# -- Insert markers around person and need spans --
def test_insert_markers_preserves_exact_text_when_person_precedes_need():
    text = "Jane needs support"
    person = make_span("person-1", "Jane", 0, 4, "person_name")
    need = make_span("need-1", "support", 11, 18, "physical_health")

    assert RelationClassifierPipeline._insert_markers(text, need, person) == (
        "[P_START]Jane[P_END] needs [N_START]support[N_END]"
    )


def test_insert_markers_preserves_exact_text_when_need_precedes_person():
    text = "Support is needed for Jane"
    need = make_span("need-1", "Support", 0, 7, "physical_health")
    person = make_span("person-1", "Jane", 22, 26, "person_name")

    assert RelationClassifierPipeline._insert_markers(text, need, person) == (
        "[N_START]Support[N_END] is needed for [P_START]Jane[P_END]"
    )


# Crossing spans (e.g. jane has dementia: [jane has], [has dementia]) cannot be represented by nested marker tokens without
# duplicating or splitting source text. Contained/nested spans (e.g. [jane has dementia], [jane], [dementia]) are supported.


def test_insert_markers_preserves_exact_text_when_nested_spans():
    text = "Jane Doe"
    person = make_span("person-1", "Jane Doe", 0, 8, "person_name")
    need = make_span("need-1", "Doe", 5, 8, "physical_health")

    marked = RelationClassifierPipeline._insert_markers(text, need, person)

    assert marked == "[P_START]Jane [N_START]Doe[N_END][P_END]"


def test_insert_markers_rejects_crossing_spans():
    text = "abcdefghij"
    need = make_span("need-1", "cdefgh", 2, 8, "physical_health")
    person = make_span("person-1", "fghij", 5, 10, "person_name")

    with pytest.raises(ValueError, match="overlap without containment"):
        RelationClassifierPipeline._insert_markers(text, need, person)


def test_predict_pairs_skips_a_crossing_pair_safely():
    pipeline = make_relation_pipeline([[0.1, 3.0]])
    text = "abcdefghij"
    crossing_need = make_span("need-cross", "cdefgh", 2, 8, "physical_health")
    crossing_person = make_span("person-cross", "fghij", 5, 10, "person_name")
    kept_need = make_span("need-kept", "ab", 0, 2, "physical_health")
    kept_person = make_span("person-kept", "ij", 8, 10, "person_name")

    relations = pipeline.predict_pairs(
        [(text, crossing_need, crossing_person), (text, kept_need, kept_person)]
    )

    assert [(relation.from_id, relation.to) for relation in relations] == [
        ("need-kept", "person-kept")
    ]


# -- Relation scoring and filtering --
def test_predict_pairs_uses_highest_probability_across_stride_windows():
    pipeline = make_relation_pipeline([[2.0, 1.0], [0.1, 3.0]])
    need = make_span("need-1", "support", 0, 7, "physical_health")
    person = make_span("person-1", "Jane", 12, 16, "person_name")

    relations = pipeline.predict_pairs([("support for Jane", need, person)])

    assert len(relations) == 1
    assert relations[0].from_id == "need-1"
    assert relations[0].to == "person-1"
    assert relations[0].confidence > 0.9


def test_predict_pairs_prunes_candidate_below_conf_threshold():
    pipeline = make_relation_pipeline([[3.0, 0.1]], threshold=0.5)
    need = make_span("need-1", "support", 0, 7, "physical_health")
    person = make_span("person-1", "Jane", 12, 16, "person_name")

    assert pipeline.predict_pairs([("support for Jane", need, person)]) == []


def test_predict_pairs_returns_one_positive_relation_for_two_candidates():
    class FakeBatch(dict):
        def __init__(self):
            super().__init__(
                input_ids=torch.ones(2, 4, dtype=torch.long),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
                overflow_to_sample_mapping=torch.arange(2),
            )

        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            return FakeBatch()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None):
            return type("Output", (), {"logits": torch.tensor([[0.1, 2.0], [2.0, 0.1]])})()

    pipeline = RelationClassifierPipeline(
        model_dir="mock/path",
        max_length=32,
        stride=8,
        threshold=0.5,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    need = make_span("need-1", "hello", 0, 5, "physical_health")
    person = make_span("person-1", "world", 6, 11, "person_name")
    relations = pipeline.predict_pairs(
        [("hello world", need, person), ("hello world", need, person)]
    )

    assert len(relations) == 1
    assert relations[0].from_id == "need-1"
