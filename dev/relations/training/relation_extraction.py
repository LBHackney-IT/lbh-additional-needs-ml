"""
Relation Extraction — Needs to Person.

Binary sequence classification to determine if a specific 'need' span applies
to a specific 'person' span. Relies on upstream models to provide the entity
boundaries. Uses entity marker injection ([N_START]/[N_END], [P_START]/[P_END])
to guide the attention mechanism toward the spans being related.
"""
from pathlib import Path
import os

import random
import json
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from common.paths import MODELS, PROCESSED, TRAIN_DATA, VAL_DATA
from common.logging import setup_logger, FileLogCallback
from common.json_helpers import save_json
from relations.relation_model import insert_markers, SPECIAL_TOKENS


# --- Config ---
@dataclass
class TrainingConfig:
    base_model: str = "albert-base-v2"
    learning_rate: float = 1e-5
    train_batch_size: int = 4
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    epochs: int = 5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 256
    logging_steps: int = 200
    early_stopping_patience: int = 2
    fp16: bool = True
    seed: int = 42


@dataclass
class Config:
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    # 1. Look for SageMaker data channels, fallback to local PROCESSED dir
    train_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", PROCESSED))
    val_dir = Path(os.environ.get("SM_CHANNEL_VAL", PROCESSED))
    
    train_path = TRAIN_DATA
    val_path = VAL_DATA

    training: TrainingConfig = field(default_factory=TrainingConfig)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    def __post_init__(self):
        random.seed(self.training.seed)
        np.random.seed(self.training.seed)
        torch.manual_seed(self.training.seed)
        torch.cuda.manual_seed_all(self.training.seed)

        # 2. Look for SageMaker's designated output directory, fallback to local MODELS dir
        sm_model_dir = os.environ.get("SM_MODEL_DIR")
        if sm_model_dir:
            self.model_dir = Path(sm_model_dir)
        else:
            self.model_dir = MODELS / "needs-relation-classifier" / self.run_name

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            f"training.relation.{self.run_name}",
            f"train_relation_{self.run_name}.log"
        )

    def save_training_params(self):
        save_json(path=self.model_dir / "config.json", data=asdict(self.training), logger=self.logger)


def build_re_dataset(path, logger) -> list[dict]:
    """Build every (need, person) pair per document, labeling each pair 1 if
    a relation links them, 0 otherwise. IDs are cast to stripped strings for
    exact matching."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for doc in raw:
        text = doc["text"]
        needs = doc.get("needs", [])
        people = doc.get("persons", [])

        valid_relations = set()
        for rel in doc.get("relations", []):
            rel_from = str(rel["from"]).strip()
            rel_to = str(rel["to"]).strip()
            
            valid_relations.add((rel_from, rel_to))

        for need in needs:
            for person in people:
                need_id = str(need["id"]).strip()
                person_id = str(person["id"]).strip()
                
                # Strictly require Need -> Person
                is_linked = 1 if (need_id, person_id) in valid_relations else 0
                marked_text = insert_markers(text, need, person)

                records.append({
                    "doc_id": doc["id"],
                    "text": marked_text,
                    "label": is_linked,
                    "need_label": need["label"],
                })

    logger.info("Built %s (need, person) pairs from %s", len(records), path)
    return records


def make_re_weights(dataset, device, logger):
    """RE datasets are almost always overwhelmingly negative (many spans,
    few actual connections). Weight formula: total / (num_classes * count)."""
    labels = dataset["label"]
    counts = Counter(labels)
    total = len(labels)

    weight_0 = total / (2 * counts[0])
    weight_1 = total / (2 * counts[1])

    logger.info("Class 0 (No Relation) count=%s weight=%.2f", counts[0], weight_0)
    logger.info("Class 1 (Relation) count=%s weight=%.2f", counts[1], weight_1)

    return torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)


class WeightedRETrainer(Trainer):
    def __init__(self, *args, loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.loss_weights)
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Binary classification metrics, reported for the RELATION class (label=1)
    specifically — given severe class imbalance, this is what actually matters,
    not overall accuracy or loss."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    config = Config()
    torch.manual_seed(config.training.seed)

    config.logger.info("Beginning training script - Relation Extraction (run=%s)", config.run_name)
    config.save_training_params()

    config.logger.info("Initializing Relation Extractor on %s...", config.device)

    train_records = build_re_dataset(config.train_path, config.logger)
    val_records = build_re_dataset(config.val_path, config.logger)

    tokenizer = AutoTokenizer.from_pretrained(config.training.base_model, add_prefix_space=False)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=False, max_length=config.training.max_length
        )

    train_ds = Dataset.from_list(train_records).map(
        tokenize_batch, batched=True, remove_columns=["text", "doc_id", "need_label"], desc="Tokenising train"
    )
    val_ds = Dataset.from_list(val_records).map(
        tokenize_batch, batched=True, remove_columns=["text", "doc_id", "need_label"], desc="Tokenising val"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.training.base_model,
        num_labels=2,
        id2label={0: "NO_RELATION", 1: "RELATION"},
        label2id={"NO_RELATION": 0, "RELATION": 1},
    )
    # CRITICAL: special tokens were added, so embeddings must be resized before training
    model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir=config.model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=config.training.train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        per_device_eval_batch_size=config.training.eval_batch_size,
        fp16=config.training.fp16 and torch.cuda.is_available(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.epochs,
        logging_steps=config.training.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=config.training.seed,
    )

    weights = make_re_weights(train_ds, config.device, config.logger)

    trainer = WeightedRETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        loss_weights=weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience),
            FileLogCallback(config.logger),
        ],
    )

    config.logger.info("Training Relation Extractor on %s...", config.device)
    trainer.train()

    best_f1 = max((h.get("eval_f1", 0.0) for h in trainer.state.log_history), default=0.0)
    config.logger.info("Best eval f1 (RELATION class): %.4f", best_f1)

    final_dir = config.model_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    config.logger.info("Saving Relation Extractor to %s...", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config.save_training_params()
    config.logger.info("Pipeline complete (run=%s). Saved to %s", config.run_name, final_dir)


if __name__ == "__main__":
    main()