import torch

from src.classifiers.relation_classifier import RelationClassifierPipeline, insert_markers
from src.common.schemas import Span


def make_span(span_id, text, start, end, label):
    return Span(id=span_id, text=text, start=start, end=end, label=label, confidence=0.9)


def make_pipeline(window_logits, threshold=0.5):
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

    pipeline = RelationClassifierPipeline.__new__(RelationClassifierPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.max_length = 32
    pipeline.stride = 8
    pipeline.threshold = threshold
    pipeline.tokenizer = FakeTokenizer()
    pipeline.model = FakeModel()
    return pipeline


def test_predict_pairs_returns_empty_for_no_candidates():
    pipeline = RelationClassifierPipeline.__new__(RelationClassifierPipeline)

    assert pipeline.predict_pairs([]) == []


def test_build_candidate_pairs_returns_cartesian_product():
    need_one = make_span("need-1", "support", 0, 7, "physical_health")
    need_two = make_span("need-2", "housing", 8, 15, "housing")
    person = make_span("person-1", "Jane", 20, 24, "person_name")

    pairs = RelationClassifierPipeline._build_candidate_pairs(
        "support housing for Jane", [need_one, need_two], [person]
    )

    assert [(need.id, person.id) for _, need, person in pairs] == [
        ("need-1", "person-1"),
        ("need-2", "person-1"),
    ]


def test_insert_markers_preserves_exact_text_when_person_precedes_need():
    text = "Jane needs support"
    person = make_span("person-1", "Jane", 0, 4, "person_name")
    need = make_span("need-1", "support", 11, 18, "physical_health")

    assert insert_markers(text, need, person) == (
        "[P_START]Jane[P_END] needs [N_START]support[N_END]"
    )


def test_insert_markers_preserves_exact_text_when_need_precedes_person():
    text = "Support is needed for Jane"
    need = make_span("need-1", "Support", 0, 7, "physical_health")
    person = make_span("person-1", "Jane", 22, 26, "person_name")

    assert insert_markers(text, need, person) == (
        "[N_START]Support[N_END] is needed for [P_START]Jane[P_END]"
    )


def test_predict_pairs_uses_highest_probability_across_overflow_windows():
    pipeline = make_pipeline([[2.0, 1.0], [0.1, 3.0]])
    need = make_span("need-1", "support", 0, 7, "physical_health")
    person = make_span("person-1", "Jane", 20, 24, "person_name")

    relations = pipeline.predict_pairs([("support for Jane", need, person)])

    assert len(relations) == 1
    assert relations[0].from_id == "need-1"
    assert relations[0].to == "person-1"

    assert relations[0].to == "person-1"
    assert relations[0].confidence > 0.9


def test_predict_pairs_filters_candidate_below_threshold():
    pipeline = make_pipeline([[3.0, 0.1]], threshold=0.5)
    need = make_span("need-1", "support", 0, 7, "physical_health")
    person = make_span("person-1", "Jane", 20, 24, "person_name")

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

    pipeline = RelationClassifierPipeline.__new__(RelationClassifierPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.max_length = 32
    pipeline.stride = 8
    pipeline.threshold = 0.5
    pipeline.tokenizer = FakeTokenizer()
    pipeline.model = FakeModel()

    need = make_span("need-1", "hello", 0, 5, "physical_health")
    person = make_span("person-1", "world", 6, 11, "person_name")
    relations = pipeline.predict_pairs(
        [("hello world", need, person), ("hello world", need, person)]
    )

    assert len(relations) == 1
    assert relations[0].from_id == "need-1"
