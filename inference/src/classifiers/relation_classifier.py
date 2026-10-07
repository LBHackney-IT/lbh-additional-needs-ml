"""
Relation classifier model and production wrapper/interface.

Binary sequence classifier over a note with entity-marker tokens injected
around one (need, person) pair.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.json_helpers import load_json
from src.common.schemas import Relation, Span

DEFAULT_MODEL_ROOT = Path("/opt/ml/processing/model")
SPECIAL_TOKENS = ["[N_START]", "[N_END]", "[P_START]", "[P_END]"]
# Tuple alias representing: (note_text, need_span, person_span).
CandidatePair = tuple[str, Span, Span]


# --- Helpers ---
def insert_markers(text: str, need: Span, person: Span) -> str:
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
    spans = sorted(
        [
            (need.start, need.end, "[N_START]", "[N_END]"),
            (person.start, person.end, "[P_START]", "[P_END]"),
        ],
        key=lambda x: x[0],
        reverse=True,
    )

    marked_text = text
    for start, end, t_start, t_end in spans:
        marked_text = (
            marked_text[:start] + t_start + marked_text[start:end] + t_end + marked_text[end:]
        )
    return marked_text


# --- Classes ---
@dataclass
class RelationClassifierPipeline:
    """Production runtime pipeline for pair-wise relation classification.

    Loads ALBERT sequence classification weights and predicts binary relation probability
    between candidate need and person spans.

    Args:
        model_dir: Path to directory containing model weights and configs.
        max_length: Maximum sequence length per window.
        stride: Overlap token length for sliding window chunking.
        threshold: Minimum confidence score required to form a valid relation.
        device: Torch execution device (CPU, CUDA, or MPS).
    """

    model_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RELATION_MODEL_DIR", DEFAULT_MODEL_ROOT / "relation" / "final_model")
        )
    )
    max_length: int | None = None
    stride: int = 128
    threshold: float = 0.5
    device: torch.device | None = None

    def __post_init__(self):
        """Initialize tokenizer, classification model, sequence limits, and hardware device."""
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
        if self.max_length is None:
            run_config = load_json(self.model_dir.parent / "config.json")
            self.max_length = run_config["max_length"]
        # Tokenizer and weights
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, add_prefix_space=False)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _build_candidate_pairs(
        text: str, needs: list[Span], persons: list[Span]
    ) -> list[CandidatePair]:
        """Generate Cartesian product of candidate need and person span pairs.

        Args:
            text: Raw case note text string.
            needs: Extracted need spans.
            persons: Extracted person spans.

        Returns:
            List of (text, need_span, person_span) tuples for relation scoring.
        """
        return [(text, need, person) for need in needs for person in persons]

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

        # Add markers for each person x need pair
        marked_texts = [
            insert_markers(text, need, person) for text, need, person in candidate_pairs
        ]

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

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            window_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

        # Aggregate window probabilities per candidate pair by taking the maximum score across chunks
        pair_max_probs: list[float] = [0.0] * len(candidate_pairs)
        for pair_idx, prob in zip(pair_indices, window_probs, strict=True):
            pair_max_probs[pair_idx] = max(pair_max_probs[pair_idx], prob)

        # Filter by threshold and format
        relations: list[Relation] = []
        for (_, need, person), prob in zip(candidate_pairs, pair_max_probs, strict=True):
            if prob > self.threshold:
                relations.append(
                    Relation(
                        from_id=need.id,
                        to=person.id,
                        confidence=float(prob),
                    )
                )

        return relations
