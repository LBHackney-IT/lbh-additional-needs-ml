"""
Shared span-classifier pieces used by both train_span_v3.py and eval_span_v3.py,
so the model architecture and candidate-generation logic can't drift between
training and evaluation.
"""

from collections import defaultdict
import uuid

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel

from common.json_helpers import load_json

def spans_overlap(a_start, a_end, b_start, b_end):
    """True if spans overlap (excluding adjacency)."""
    return max(a_start, b_start) < min(a_end, b_end)

class SpanClassifier(nn.Module):
    def __init__(self, base_model: str, num_labels: int, class_weight=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model)
        self.classifier = nn.Linear(self.backbone.config.hidden_size * 2, num_labels + 1)
        self.class_weight = class_weight

    def forward(self, input_ids, attention_mask, candidate_spans, labels=None):
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if candidate_spans.shape[1] == 3:
            # Batched path: columns are (batch_idx, start, end)
            batch_idx, starts, ends = candidate_spans[:, 0], candidate_spans[:, 1], candidate_spans[:, 2]
            start_vecs = hidden_states[batch_idx, starts]
            end_vecs = hidden_states[batch_idx, ends - 1]
        else:
            # Legacy single-example path: columns are (start, end), batch size 1
            h = hidden_states[0]
            start_vecs = h[candidate_spans[:, 0]]
            end_vecs = h[candidate_spans[:, 1] - 1]

        logits = self.classifier(torch.cat([start_vecs, end_vecs], dim=1))
        loss = nn.CrossEntropyLoss(weight=self.class_weight)(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def generate_candidates(offsets, max_size: int):
    """Return (start_idx, end_idx) candidates over real token positions only."""
    real_positions = [i for i, (s, e) in enumerate(offsets) if s != e]
    candidates = []
    for size in range(1, max_size + 1):
        for i in range(len(real_positions) - size + 1):
            window = real_positions[i:i + size]
            if window[-1] - window[0] == size - 1:
                candidates.append((window[0], window[-1] + 1))
    return candidates


def correct_offset(text: str, char_start: int, char_end: int) -> tuple:
    span_text = text[char_start:char_end]
    n_stripped = len(span_text) - len(span_text.lstrip())
    return char_start + n_stripped, char_end


def run_inference(texts, model, tokenizer, config):
    """
    Batched inference. `texts` may be a single string or a list of strings;
    always returns per-example results as parallel lists.
    """
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    all_offsets = tokenized["offset_mapping"].tolist()

    # Generate candidates per example, flattened with a batch index
    per_example_candidates = []
    flat_candidates = []
    for b, offsets in enumerate(all_offsets):
        candidates = generate_candidates(offsets, config.max_candidate_size)
        per_example_candidates.append(candidates)
        flat_candidates.extend((b, start, end) for start, end in candidates)

    if not flat_candidates:
        empty = [None] * len(texts)
        results = [[] for _ in texts]
        return (results[0], empty[0]) if single_input else (results, empty)

    input_ids = tokenized["input_ids"].to(config.device)
    attention_mask = tokenized["attention_mask"].to(config.device)
    candidate_spans = torch.tensor(flat_candidates, dtype=torch.long, device=config.device)

    outputs = model(input_ids, attention_mask, candidate_spans)
    all_probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

    # Split flat outputs back into per-example slices
    correct_leading_whitespace_offset = "deberta" in config.base_model.lower()
    results_spans, results_probs = [], []
    cursor = 0
    for text, offsets, candidates in zip(texts, all_offsets, per_example_candidates):
        n = len(candidates)
        if n == 0:
            results_spans.append([])
            results_probs.append(None)
            continue

        example_probs = all_probs[cursor:cursor + n]
        cursor += n

        char_spans = []
        for tok_start, tok_end in candidates:
            char_start = offsets[tok_start][0]
            char_end = offsets[tok_end - 1][1]
            if correct_leading_whitespace_offset:
                char_start, char_end = correct_offset(text, char_start, char_end)
            char_spans.append((char_start, char_end))

        results_spans.append(char_spans)
        results_probs.append(example_probs)

    return (results_spans[0], results_probs[0]) if single_input else (results_spans, results_probs)


def deduplicate_predictions(pred_list):
    """Keep the highest-confidence prediction when spans overlap for the same label.
    pred_spans: dict of (id, text, start, end, label, confidence)."""
    if not pred_list:
        return []

    # Greedy highest-confidence first keeping, per label
    sorted_preds = sorted(pred_list, key=lambda item: item["confidence"], reverse=True)
    kept = []
    for pred in sorted_preds:
        if not any(
            pred["label"] == k["label"] and spans_overlap(pred["start"], pred["end"], k["start"], k["end"])
            for k in kept
        ):
            kept.append(pred)
    return kept

def threshold_and_format(text, char_spans, probs, label_list, thresholds):
    if probs is None or not char_spans:
        return []

    # Take most likely class & confidence
    no_entity_id = len(label_list)
    pred_classes = np.argmax(probs, axis=-1)
    pred_confs = np.max(probs, axis=-1)

    # Set up thresholds (force 1.0 for no_entity)
    thresh_arr = np.array(
        [thresholds.get(label_list[c], 0.5) if c != no_entity_id else 1.0 for c in pred_classes]
    )

    # Keep candidate if not predicted no_entity and clears threshold
    pred_span_list = [
        {
            "id": str(uuid.uuid4())[:8],
            "text": text[char_spans[i][0]:char_spans[i][1]],
            "start": char_spans[i][0],
            "end": char_spans[i][1],
            "label": label_list[pred_classes[i]],
            "confidence": float(pred_confs[i]),
        }
        for i in range(len(pred_classes))
        if pred_classes[i] != no_entity_id and pred_confs[i] >= thresh_arr[i]
    ]

    return deduplicate_predictions(pred_span_list)


def load_thresholds(config, label_list):
    if config.thresholds_path.exists():
        thresholds = load_json(config.thresholds_path, config.logger)
        config.logger.info("Loaded pre-fit thresholds from %s", config.thresholds_path)
        return thresholds

    config.logger.warning(
        "No thresholds file found at %s — defaulting to 0.5 for all %d classes",
        config.thresholds_path,
        len(label_list),
    )
    return {label: 0.5 for label in label_list}