"""
Custom span classifier model and production wrapper/interface.

Architecture: a transformer backbone plus a linear head over [start_token; end_token]
concatenations.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

# --- Constants ---
DEFAULT_MODEL_ROOT = Path("/opt/ml/processing/model")


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
