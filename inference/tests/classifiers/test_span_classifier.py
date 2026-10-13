import torch

from src.classifiers.span_classifier import SpanClassifier


class DummySpanBackbone(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids=None, attention_mask=None):
        batch, seq = input_ids.shape
        hidden = torch.randn(batch, seq, self.config.hidden_size)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_span_classifier_returns_candidate_logits_and_no_entity_class(monkeypatch):
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


def test_span_classifier_returns_loss_when_labels_are_supplied(monkeypatch):
    monkeypatch.setattr(
        "src.classifiers.span_classifier.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: DummySpanBackbone(hidden_size=16),
    )
    model = SpanClassifier("dummy-model", num_labels=3)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    candidate_spans = torch.tensor([[0, 1, 3]])
    labels = torch.tensor([1])

    output = model(input_ids, attention_mask, candidate_spans, labels=labels)
    assert output["loss"].ndim == 0
