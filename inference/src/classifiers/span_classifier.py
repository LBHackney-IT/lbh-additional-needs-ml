"""
Span classifier model and production wrapper/interface.

Architecture matches the training script (`dev/spans/span_model.py`):
a transformer backbone plus a linear head over [start_token; end_token]
concatenations.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from src.common.json_helpers import load_json
from src.common.schemas import Span, SpanPredictions

# --- Constants ---
DEFAULT_MODEL_ROOT = Path("/opt/ml/processing/model")


# --- Helpers ---
def load_thresholds(
    thresholds_path: Path | None,
    label_list: list[str],
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """Load classification thresholds per class or default to 0.5 fallback.

    Args:
        thresholds_path: Path to optimized thresholds JSON file.
        label_list: List of valid class labels.
        logger: Logger instance for audit logging.

    Returns:
        Mapping of label strings to threshold values.
    """
    if thresholds_path and thresholds_path.exists():
        thresholds = load_json(thresholds_path, logger)
        if logger:
            logger.info("Loaded pre-fit thresholds from %s", thresholds_path)
        return thresholds

    if logger:
        logger.warning(
            "No thresholds file found at %s — defaulting to 0.5 for all %d classes",
            thresholds_path,
            len(label_list),
        )
    return {label: 0.5 for label in label_list}


def generate_candidates(offsets: list[tuple[int, int]], max_size: int) -> list[tuple[int, int]]:
    """Enumerate token spans using a sliding window over non-special tokens.

    Args:
        offsets: List of (char_start, char_end) offsets for a tokenized sequence.
        max_size: Maximum window length for candidate spans.

    Returns:
        List of candidate token index pairs (token_start, token_end).
    """
    real_positions = [i for i, (s, e) in enumerate(offsets) if s != e]
    candidates = []
    for size in range(1, max_size + 1):
        for i in range(len(real_positions) - size + 1):
            window = real_positions[i : i + size]
            # Do not bridge gaps introduced by special or padding tokens.
            if window[-1] - window[0] == size - 1:
                candidates.append((window[0], window[-1] + 1))
    return candidates


def correct_offset(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Strip leading whitespace introduced by DeBERTa tokenizer offset mappings.

    Args:
        text: Original input document string.
        char_start: Uncorrected character start offset.
        char_end: Character end offset.

    Returns:
        Adjusted (char_start, char_end) pair with leading whitespace removed.
    """
    span_text = text[char_start:char_end]
    n_stripped = len(span_text) - len(span_text.lstrip())
    return char_start + n_stripped, char_end


def deduplicate_predictions(pred_list: list[Span]) -> list[Span]:
    """Perform greedy non-maximum suppression on predicted spans.

    Suppresses lower-confidence overlapping spans only when they share the same label.

    Args:
        pred_list: List of span prediction dictionaries with keys 'label', 'confidence',
            'start', and 'end'.

    Returns:
        Deduplicated list of prediction dictionaries.
    """
    if not pred_list:
        return []

    def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        """Check if spans [a_start, a_end) and [b_start, b_end) overlap."""
        return max(a_start, b_start) < min(a_end, b_end)

    sorted_preds = sorted(pred_list, key=lambda item: item.confidence, reverse=True)
    kept: list[Span] = []
    for pred in sorted_preds:
        # Different labels may overlap; suppress only same-label duplicates.
        if not any(
            pred.label == kept_span.label
            and spans_overlap(pred.start, pred.end, kept_span.start, kept_span.end)
            for kept_span in kept
        ):
            kept.append(pred)
    return kept


def threshold_and_format(
    text: str,
    char_spans: list[tuple[int, int]],
    probs: np.ndarray,
    label_list: list[str],
    thresholds: dict[str, float],
) -> list[Span]:
    """Filter predictions by confidence thresholds and format into output schema.

    Args:
        text: Original document text string.
        char_spans: List of (start_char, end_char) boundaries matching probabilities.
        probs: Array of class probabilities of shape (num_spans, num_classes + 1).
        label_list: Ordered list of entity class names.
        thresholds: Mapping of class names to calibrated minimum confidence thresholds.

    Returns:
        List of formatted entity prediction dicts.
    """
    if probs is None or not char_spans:
        return []

    no_entity_id = len(label_list)
    pred_classes = np.argmax(probs, axis=-1)
    pred_confs = np.max(probs, axis=-1)
    thresh_arr = np.array(
        [thresholds.get(label_list[c], 0.5) if c != no_entity_id else 1.0 for c in pred_classes]
    )

    pred_span_list = [
        Span(
            id=str(uuid.uuid4())[:8],
            text=text[char_spans[i][0] : char_spans[i][1]],
            start=char_spans[i][0],
            end=char_spans[i][1],
            label=label_list[pred_classes[i]],
            confidence=float(pred_confs[i]),
        )
        for i in range(len(pred_classes))
        if pred_classes[i] != no_entity_id and pred_confs[i] >= thresh_arr[i]
    ]

    return pred_span_list


# --- Classes ---
class SpanClassifier(nn.Module):
    """Transformer-based classifier for candidate NER span classification.

    Attributes:
        backbone: Pre-trained HuggingFace transformer model.
        classifier: Linear head mapping concatenated boundary tokens [start, end) to logits.
        class_weight: Optional class weights for loss computation.
    """

    def __init__(self, base_model: str, num_labels: int, class_weight: torch.Tensor | None = None):
        """Initialize the span classifier.

        Args:
            base_model: Name or path of the HuggingFace base transformer.
            num_labels: Number of valid target entity categories (number of AN labels and person labels excluding 'no entity').
            class_weight: Optional cross-entropy loss weights for class imbalance.
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model)
        self.classifier = nn.Linear(self.backbone.config.hidden_size * 2, num_labels + 1)
        self.class_weight = class_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_spans: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode every note once, then compute logits and (optionally) loss each proposed token span.

        Args:
            input_ids: Batched input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            candidate_spans: Tensor of shape (num_candidates, 3) containing
                (batch_index, token_start, token_end) indices.
            labels: Optional ground truth labels for loss calculation.

        Returns:
            Dictionary containing 'logits' and optionally 'loss'.
        """
        # 1. Run through transformer backbone to provide contextual embeddings to each token
        hidden_states = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # 2. Concatenate start + end embeddings
        batch_idx, starts, ends = (
            candidate_spans[:, 0],
            candidate_spans[:, 1],
            candidate_spans[:, 2],
        )
        start_vecs = hidden_states[batch_idx, starts]
        end_vecs = hidden_states[batch_idx, ends - 1]

        # 3. Apply classification head, return logits and optionally loss.
        logits = self.classifier(torch.cat([start_vecs, end_vecs], dim=1))
        loss = (
            nn.CrossEntropyLoss(weight=self.class_weight)(logits, labels)
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


@dataclass
class SpanClassifierPipeline:
    """Production runtime pipeline for batched span NER.

    Handles model weights initialization, tokenization, batch candidate generation, and output formatting.

    Args:
        model_dir: Path to directory containing model weights and configs.
        person_labels: Class labels designated as person entities.
        device: Torch execution device (CPU, CUDA, or MPS).
        logger: Logger instance for telemetry.
        stride: Overlap token length for sliding window chunking.
    """

    model_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SPAN_MODEL_DIR", DEFAULT_MODEL_ROOT / "span" / "final_model")
        )
    )
    person_labels: list[str] = field(default_factory=lambda: ["person_role", "person_name"])
    device: torch.device | None = None
    logger: logging.Logger | None = None
    stride: int = 128

    def __post_init__(self):
        """Initialize models, tokenizers, configurations, and the hardware acceleration."""
        # Choose accelerator
        self.device = self.device or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        # Training run config
        self.model_dir = Path(self.model_dir)
        run_config = load_json(self.model_dir.parent / "config.json", self.logger)
        self.base_model = run_config["base_model"]
        self.max_length = run_config["max_length"]
        self.max_candidate_size = run_config["max_candidate_size"]
        # Labels and thresholds
        self.label_list = load_json(self.model_dir / "label_list.json", self.logger)
        self.thresholds = load_thresholds(
            self.model_dir / "optimized_thresholds.json",
            self.label_list,
            self.logger,
        )
        # Tokenizer and weights
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = SpanClassifier(self.base_model, num_labels=len(self.label_list)).to(
            self.device
        )
        state_dict = load_file(self.model_dir / "model.safetensors", device=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Disable training behaviour

    def predict_batch(self, texts: list[str]) -> list[SpanPredictions]:
        """Run batch span prediction over a list of document texts.

        Args:
            texts: List of input document strings.

        Returns:
            Typed span predictions grouped into needs and persons for each document.
        """
        if not texts:
            return []

        # 1. Split each document into overlapping windows of up to `max_length` tokens. (Overlap = stride)
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        # Store which windows came from which text so offsets can be linked back
        sample_mapping = tokenized["overflow_to_sample_mapping"].tolist()
        all_offsets = tokenized["offset_mapping"].tolist()

        # 2. For each window, generate & list every candidate span as (token_start, token_end).
        per_window_candidates = []
        flat_candidates = []
        for chunk_idx, offsets in enumerate(all_offsets):
            candidates = generate_candidates(offsets, self.max_candidate_size)
            per_window_candidates.append(candidates)
            flat_candidates.extend((chunk_idx, start, end) for start, end in candidates)

        if not flat_candidates:
            return [SpanPredictions() for _ in texts]

        # 3. Classify every candidate span in every window in a single forward pass.
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        candidate_spans = torch.tensor(flat_candidates, dtype=torch.long, device=self.device)

        outputs = self.model(input_ids, attention_mask, candidate_spans)
        all_probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

        # 4. Walk through the windows in order. Convert each window's token spans to character positions, grouped by original document.
        correct_leading_whitespace_offset = "deberta" in self.base_model.lower()

        doc_spans: list[list[Span]] = [[] for _ in texts]
        cursor = 0
        for _chunk_idx, (doc_idx, offsets, candidates) in enumerate(
            zip(sample_mapping, all_offsets, per_window_candidates, strict=True)
        ):
            n = len(candidates)
            if n == 0:
                continue

            chunk_probs = all_probs[cursor : cursor + n]
            cursor += n

            # Get character indices from tokens
            raw_text = texts[doc_idx]
            char_spans = []
            for tok_start, tok_end in candidates:
                char_start = offsets[tok_start][0]
                char_end = offsets[tok_end - 1][1]
                if correct_leading_whitespace_offset:
                    char_start, char_end = correct_offset(raw_text, char_start, char_end)
                char_spans.append((char_start, char_end))

            # Format spans for this chunk
            chunk_extracted = threshold_and_format(
                raw_text, char_spans, chunk_probs, self.label_list, self.thresholds
            )
            doc_spans[doc_idx].extend(chunk_extracted)

        # 5. Deduplicate labels globally per doc & format
        results: list[SpanPredictions] = []
        for _raw_text, spans in zip(texts, doc_spans, strict=True):
            deduped = deduplicate_predictions(spans)
            results.append(
                SpanPredictions(
                    needs=[s for s in deduped if s.label not in self.person_labels],
                    persons=[s for s in deduped if s.label in self.person_labels],
                )
            )

        return results
