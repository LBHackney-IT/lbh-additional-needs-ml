"""
Runs inference and sweeps per-class thresholds to find the champions.
Saves optimised thresholds.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm

from common.paths import VAL_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.evaluators import SpanEvaluator
from spans.span_model import SpanClassifier, generate_candidates, spans_overlap


@dataclass
class Config:
    model_dir: Path
    val_path: Path = VAL_DATA
    thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.run_name = self.model_dir.parent.name
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.logger = setup_logger(
            f"optim_threshold.span.{self.run_name}",
            f"optim_threshold_span_{self.run_name}.log",
        )

        self.thresholds_path: Path = self.model_dir / "optimized_thresholds.json"
        
        # Training params
        config_path = self.model_dir.parent / "config.json"
        run_config = load_json(config_path, self.logger)
        self.base_model = run_config["base_model"]
        self.max_length = run_config["max_length"]
        self.max_candidate_size = run_config["max_candidate_size"]


def correct_offset(text: str, char_start: int, char_end: int) -> tuple:
    span_text = text[char_start:char_end]
    n_stripped = len(span_text) - len(span_text.lstrip())
    return char_start + n_stripped, char_end


def run_inference(text, model, tokenizer, config):
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = tokenized["offset_mapping"][0].tolist()
    candidates = generate_candidates(offsets, config.max_candidate_size)
    if not candidates:
        return [], None

    input_ids = tokenized["input_ids"].to(config.device)
    attention_mask = tokenized["attention_mask"].to(config.device)
    candidate_spans = torch.tensor(candidates, dtype=torch.long, device=config.device)

    outputs = model(input_ids, attention_mask, candidate_spans)
    probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

    char_spans = []
    correct_leading_whitespace_offset = "deberta" in config.base_model.lower()
    for tok_start, tok_end in candidates:
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end - 1][1]

        if correct_leading_whitespace_offset:
            char_start, char_end = correct_offset(text, char_start, char_end)

        char_spans.append((char_start, char_end))

    return char_spans, probs


def deduplicate_predictions(pred_list):
    """Keep the highest-confidence prediction when spans overlap for the same label.
    pred_spans: list of tuples: (start, end, label, confidence)."""
    if not pred_list:
        return []

    # Greedy highest-confidence first keeping, per label
    sorted_preds = sorted(pred_list, key=lambda item: item[3], reverse=True)
    kept = []
    for start, end, label, conf in sorted_preds:
        if not any(
            label == k_label and spans_overlap(start, end, k_start, k_end)
            for k_start, k_end, k_label, _ in kept
        ):
            kept.append((start, end, label, conf))
    return kept


def threshold_to_spans(char_spans, probs, label_list, threshold):
    if probs is None or not char_spans:
        return []

    background_id = len(label_list)
    pred_classes = np.argmax(probs, axis=-1)
    pred_confs = np.max(probs, axis=-1)

    if isinstance(threshold, dict):
        thresh_arr = np.array(
            [threshold.get(label_list[c], 0.5) if c != background_id else 1.0 for c in pred_classes]
        )
    else:
        thresh_arr = np.full(len(pred_classes), threshold)

    pred_span_list = [
        (char_spans[i][0], char_spans[i][1], label_list[pred_classes[i]], pred_confs[i])
        for i in range(len(pred_classes))
        if pred_classes[i] != background_id and pred_confs[i] >= thresh_arr[i]
    ]

    return deduplicate_predictions(pred_span_list)


def get_class_f1(results_dict, label, match_strategy="iou_0.5"):
    try:
        return results_dict["per_label"][label][match_strategy]["f1"]
    except KeyError:
        return 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python model_optimize_thresholds.py <path/to/final_model>")
        sys.exit(1)

    config = Config(model_dir=Path(sys.argv[1]))
    config.logger.info(
        "Running inference and optimizing thresholds for %s on %s",
        config.model_dir,
        config.device,
    )
    
    # Load val data
    val_records = load_json(config.val_path, config.logger)

    # Set up model
    label_list = load_json(config.model_dir / "label_list.json", config.logger)
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = SpanClassifier(config.base_model, num_labels=len(label_list)).to(config.device)

    state_dict = load_file(config.model_dir / "model.safetensors", device=str(config.device))
    load_result = model.load_state_dict(state_dict)
    config.logger.info("Loaded model weights: %s", load_result)
    model.eval()

    # Run inference (once)
    predictions = []
    with torch.inference_mode():
        for record in tqdm(val_records):
            text = record.get("text", "")
            char_spans, probs = run_inference(text, model, tokenizer, config)
            predictions.append(
                {
                    "id": record.get("id"),
                    "char_spans": char_spans,
                    "probs": probs,
                }
            )

    # Create y_true and y_preds for comparison / eval
    pred_by_id = {p["id"]: p for p in predictions}
    y_true, y_preds, matched_records = [], [], []

    for record in val_records:
        record_id = record.get("id")
        pred = pred_by_id.get(record_id)
        if pred is None:
            config.logger.warning("No prediction found for record %s", record_id)
            continue

        true_spans = [
            (item["start"], item["end"], item["label"])
            for item in record.get("needs", []) + record.get("persons", [])
            if "label" in item
        ]
        y_true.append(true_spans)
        y_preds.append((pred["char_spans"], pred["probs"]))
        matched_records.append(record)

    config.logger.info("Loaded %d records with predictions and ground truth", len(y_true))

    # Loop through thresholds & eval
    evaluator = SpanEvaluator(label_list, config.logger)
    results_by_threshold = {}

    config.logger.info("Evaluating %d thresholds...", len(config.thresholds))
    for threshold in config.thresholds:
        y_pred = [
            [(s, e, l) for s, e, l, _ in threshold_to_spans(char_spans, probs, label_list, threshold)]
            if probs is not None
            else []
            for char_spans, probs in y_preds
        ]
        results_by_threshold[threshold] = evaluator.evaluate(y_true, y_pred)

    # Choose theeshold for each class
    config.logger.info("Finding champion threshold for each class...")
    optimized_thresholds = {}
    for label in label_list:
        best_f1, best_thresh = -1.0, 0.5
        for threshold in config.thresholds:
            class_f1 = get_class_f1(results_by_threshold[threshold], label, match_strategy="iou_0.5")
            if class_f1 > best_f1:
                best_f1, best_thresh = class_f1, threshold
        optimized_thresholds[label] = best_thresh
        config.logger.info("Class Winner | %s -> Threshold: %s (F1: %.4f)", label, best_thresh, best_f1)

    # Save out
    save_json(path=config.thresholds_path, data=optimized_thresholds, logger=config.logger)
    config.logger.info("Threshold optimization complete. Saved thresholds to %s", config.thresholds_path)


if __name__ == "__main__":
    main()