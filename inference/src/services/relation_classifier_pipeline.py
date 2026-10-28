"""
Relation classifier model and production wrapper/interface.

Binary sequence classifier over a note with entity-marker tokens injected
around one (need, person) pair.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.device import resolve_device
from src.common.json_helpers import load_json
from src.common.schemas import Relation, Span

DEFAULT_MODEL_ROOT = Path("/opt/ml/processing/model")
SPECIAL_TOKENS = ["[N_START]", "[N_END]", "[P_START]", "[P_END]"]
# Tuple alias representing: (note_text, need_span, person_span).
CandidatePair = tuple[str, Span, Span]


def _cast_for_device(model: torch.nn.Module, device: torch.device) -> None:
    """Run the relation model in float16 on CUDA.

    CPU and MPS stay in float32. Half precision is not supported there.
    """
    if device.type == "cuda":
        model.half()
    else:
        model.float()


class RelationClassifierPipeline:
    def __init__(
        self,
        model_dir: Path | str | None = None,
        max_length: int | None = None,
        stride: int = 128,
        window_batch_size: int = 128,
        threshold: float = 0.5,
        device: torch.device | None = None,
        # DEPENDENCY INJECTIONS
        tokenizer: Any | None = None,
        model: Any | None = None,
    ):
        # 1. Set simple configuration
        self.stride = stride
        if window_batch_size < 1:
            raise ValueError(f"window_batch_size must be positive, got {window_batch_size}")
        self.window_batch_size = window_batch_size
        self.threshold = threshold
        self.model_dir = Path(
            model_dir or os.environ.get("RELATION_MODEL_DIR", "DEFAULT_MODEL_ROOT")
        )

        # 2. Resolve hardware device
        self.device = resolve_device(device)

        # 3. Resolve max_length
        self.max_length = max_length
        if self.max_length is None:
            # (Assuming load_json is defined elsewhere)
            run_config = load_json(self.model_dir.parent / "config.json")
            self.max_length = run_config["max_length"]

        # 4. Inject or load heavy dependencies
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_dir, add_prefix_space=False
        )

        self.model = model
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            _cast_for_device(self.model, self.device)
            self.model.eval()

    # --- Helpers ---
    @staticmethod
    def _insert_markers(text: str, need: Span, person: Span) -> str:
        """Inject entity marker tokens around candidate need and person spans.

        Inserts tokens from right to left so character offset locations
        for earlier spans stay correct.

        Args:
            text: Raw case note document string.
            need: Need span with 'start' and 'end' character offsets.
            person: Person span with 'start' and 'end' character offsets.

        Returns:
            Document string with [N_START], [N_END], [P_START], and [P_END] markers injected.
        """
        spans = [
            (need.start, need.end, "[N_START]", "[N_END]"),
            (person.start, person.end, "[P_START]", "[P_END]"),
        ]
        for start, end, _start_marker, _end_marker in spans:
            if not 0 <= start <= end <= len(text):
                raise ValueError(f"Span offsets are outside the source text: {start}:{end}")

        # Crossing spans (e.g. jane has dementia: [jane has], [has dementia]) cannot be represented by nested marker tokens without
        # duplicating or splitting source text. Contained/nested spans (e.g. [jane has dementia], [jane], [dementia]) are supported.
        first, second = spans
        if (first[0] < second[0] < first[1] < second[1]) or (
            second[0] < first[0] < second[1] < first[1]
        ):
            raise ValueError("Need and person spans overlap without containment")

        events: dict[int, list[tuple[int, int, str]]] = {}
        for start, end, start_marker, end_marker in spans:
            # End markers are emitted before start markers at the same offset.
            # Longer spans open first; shorter spans close first.
            events.setdefault(start, []).append((1, -(end - start), start_marker))
            events.setdefault(end, []).append((0, end - start, end_marker))

        marked_parts: list[str] = []
        cursor = 0
        for position in sorted(events):
            marked_parts.append(text[cursor:position])
            marked_parts.extend(marker for _, _, marker in sorted(events[position]))
            cursor = position
        marked_parts.append(text[cursor:])
        return "".join(marked_parts)

    @staticmethod
    def _build_candidate_pairs(
        text: str, needs: list[Span], persons: list[Span]
    ) -> list[CandidatePair]:
        """Generate Cartesian product of candidate need and person span pairs.

        Property-level needs belong to the tenancy, so they are left out.

        Args:
            text: Raw case note text string.
            needs: Extracted need spans.
            persons: Extracted person spans.

        Returns:
            List of (text, need_span, person_span) tuples for relation scoring.
        """
        return [
            (text, need, person)
            for need in needs
            if not need.label.startswith("property_level")
            for person in persons
        ]

    def _positive_probabilities(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[float]:
        """Score relation windows in groups and return the positive-class probability for each.

        Windows stay in tokenizer order, so the caller can still take the max
        probability across the windows that belong to one need-person pair.
        """
        probabilities: list[float] = []
        window_count = input_ids.shape[0]
        with torch.inference_mode():
            # Chunk into windows of size self.window_batch_size
            for start in range(0, window_count, self.window_batch_size):
                end = min(start + self.window_batch_size, window_count)
                logits = self.model(
                    input_ids=input_ids[start:end].to(self.device),
                    attention_mask=attention_mask[start:end].to(self.device),
                ).logits
                probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                del logits
        return probabilities

    # --- Public ---
    def predict_from_spans(
        self, text: str, needs: list[Span], persons: list[Span]
    ) -> list[Relation]:
        """Wrapper to build candidate pairs and predict relations in one call."""
        candidate_pairs = self._build_candidate_pairs(text, needs, persons)
        return self.predict_pairs(candidate_pairs)

    def predict_pairs(self, candidate_pairs: list[CandidatePair]) -> list[Relation]:
        """Score (text, need, person) candidate tuples and return valid links above threshold.

        Args:
            candidate_pairs: List of (document_text, need_span, person_span) tuples.

        Returns:
            List of need-to-person relations.
        """
        if not candidate_pairs:
            return []

        # Crossing spans cannot take markers. Skip the pair and score the rest.
        marked_texts = []
        scoreable_pairs = []
        for text, need, person in candidate_pairs:
            try:
                marked_texts.append(self._insert_markers(text, need, person))
            except ValueError as error:
                if "overlap without containment" not in str(error):
                    raise
                continue
            scoreable_pairs.append((text, need, person))

        if not scoreable_pairs:
            return []

        # Tokenize with sliding window chunking
        tokenized = self.tokenizer(
            marked_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        pair_indices = tokenized["overflow_to_sample_mapping"].tolist()
        # One note can have hundreds of need-person pairs, and each pair splits
        # into windows. Score those windows in groups so one forward fits on the GPU.
        window_probs = self._positive_probabilities(
            tokenized["input_ids"], tokenized["attention_mask"]
        )

        # Aggregate window probabilities per candidate pair by taking the maximum score across chunks
        pair_max_probs: list[float] = [0.0] * len(scoreable_pairs)
        for pair_idx, prob in zip(pair_indices, window_probs, strict=True):
            pair_max_probs[pair_idx] = max(pair_max_probs[pair_idx], prob)

        # Filter by threshold and format
        relations: list[Relation] = []
        for (_, need, person), prob in zip(scoreable_pairs, pair_max_probs, strict=True):
            if prob > self.threshold:
                relations.append(
                    Relation(
                        from_id=need.id,
                        to=person.id,
                        confidence=float(prob),
                    )
                )

        return relations
