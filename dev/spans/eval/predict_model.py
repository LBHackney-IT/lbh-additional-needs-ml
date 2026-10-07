"""
Generate span predictions using batched `run_inference` from shared.span_model.
Saves predictions to results/predictions as `span.<run_name>.batched.json`.
"""
import sys
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import uuid

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm

from common.paths import METRICS, PREDICTIONS, VAL_DATA, TEST_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from spans.span_model import SpanClassifier, run_inference, threshold_and_format, load_thresholds


@dataclass
class Config:
    model_dir: Path
    data_path: Path = TEST_DATA
    person_labels: List[str] = field(default_factory=lambda: ["person_role", "person_name"])
    inference_batch_size: int = 8

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
            f"run_thresholded.span.{self.run_name}",
            f"run_thresholded_span_{self.run_name}.log",
        )

        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        METRICS.mkdir(parents=True, exist_ok=True)

        self.output_path: Path = PREDICTIONS / f"span.{self.run_name}.json"
        self.eval_path: Path = METRICS / f"span.{self.run_name}.json"

        self.thresholds_path: Path = self.model_dir / "optimized_thresholds.json"
        config_path = self.model_dir.parent / "config.json"
        run_config = load_json(config_path, self.logger)
        self.base_model = run_config["base_model"]
        self.max_length = run_config["max_length"]
        self.max_candidate_size = run_config["max_candidate_size"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_model_batched.py <path/to/final_model>")
        sys.exit(1)

    config = Config(sys.argv[1])
    config.logger.info("Running batched inference for %s on %s", config.model_dir, config.device)

    # Load data
    val_records = load_json(config.data_path, config.logger)

    # Load model config
    label_list = load_json(config.model_dir / "label_list.json", config.logger)
    thresholds = load_thresholds(config, label_list)

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    model = SpanClassifier(config.base_model, num_labels=len(label_list)).to(config.device)

    state_dict = load_file(config.model_dir / "model.safetensors", device=str(config.device))
    load_result = model.load_state_dict(state_dict)
    config.logger.info("Loaded model weights: %s", load_result)
    model.eval()

    # Run batched inference (chunked to control memory)
    predictions = []
    with torch.inference_mode():
        # chunk helper (simple local implementation)
        def chunked(iterable, size):
            from itertools import islice
            it = iter(iterable)
            while batch := list(islice(it, size)):
                yield batch

        for batch in tqdm(list(chunked(val_records, config.inference_batch_size)), desc="Batches"):
            texts = [r.get("text", "") for r in batch]
            ids = [r.get("id", "") for r in batch]
            try:
                batch_raw_spans, batch_probs = run_inference(texts, model, tokenizer, config)

                for record, text, raw_spans, probs in zip(batch, texts, batch_raw_spans, batch_probs):
                    clean_spans = threshold_and_format(text, raw_spans, probs, label_list, thresholds)

                    predictions.append(
                        {
                            "id": record.get("id"),
                            "text": text,
                            "date": record.get("note_date"),
                            "model": config.run_name,
                            "needs": [x for x in clean_spans if x['label'] not in config.person_labels],
                            "persons": [x for x in clean_spans if x['label'] in config.person_labels],
                            "tenure_ids": record.get("tenure_ids"),
                            "household_members": record.get("household_members"),
                        }
                    )
            except Exception as e:
                config.logger.error("Batch failed (ids %s): %s", ids, e)
            finally:
                if config.device.type == "mps":
                    torch.mps.empty_cache()

    # Save out
    save_json(path=config.output_path, data=predictions, logger=config.logger)
    config.logger.info("Generated batched predictions for %d records. Saved to %s", len(predictions), config.output_path)


if __name__ == "__main__":
    main()
