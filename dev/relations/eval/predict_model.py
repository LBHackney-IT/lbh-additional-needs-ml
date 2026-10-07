"""
Generate relation predictions from trained model.
Outputs standardized predictions for unified evaluation.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.paths import PROCESSED, PREDICTIONS, TEST_DATA
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from relations.relation_model import insert_markers, SPECIAL_TOKENS


@dataclass
class Config:
    model_dir: Path
    data_path: Path # Defaults to val_data, can be overidden with predictions
    output_dir: Path = PREDICTIONS
    output_filename: str = "relation.{model_names}.json"
    threshold: float = 0.5 # TODO: Add tuning.

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.data_path = Path(self.data_path)

        self.run_name = self.model_dir.parent.name
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.logger = setup_logger(
            f"predict.{self.run_name}_relations",
            f"predict_{self.run_name}_relations.log",
        )

        config_path = self.model_dir.parent / "config.json"
        run_config = load_json(config_path, self.logger)
        self.base_model = run_config["base_model"]
        self.max_length = run_config["max_length"]


def run_inference(
    model, tokenizer, text: str, need: dict, person: dict, device, max_length: int
) -> float:
    """Run inference for a single (need, person) pair. Returns probability of relation."""
    marked_text = insert_markers(text, need, person)
    
    inputs = tokenizer(
        marked_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        relation_prob = probs[0, 1].item()  # Probability of class 1 (relation exists)

    return relation_prob


def predict_relations(data: List[dict], config: Config) -> Dict[str, dict]:
    """Generate relation predictions from model."""
    config.logger.info(f"Using device: {config.device}")

    # Load model and tokenizer
    config.logger.info(f"Loading model from {config.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, add_prefix_space=False)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_dir).to(config.device).eval()

    predictions = []

    # Loop through docs
    for doc in tqdm(data):
        text = doc["text"]
        needs = doc.get("needs", [])
        persons = doc.get("persons", [])

        # Generate (need, person) pairs with predictions
        doc_relations = []
        for need in needs:
            # Ignore property-level labels (they never need relations)
            if need["label"].startswith("property_level"):
                continue
            for person in persons:
                prob = run_inference(
                    model, tokenizer, text, need, person, config.device, max_length=config.max_length
                )
                if prob > config.threshold:
                    doc_relations.append(
                        {
                            "from": need["id"],
                            "to": person["id"],
                            "confidence": prob
                        }
                    )

        predictions.append({
            "id": doc["id"],
            "text": text,
            "date": doc.get("date"),
            "model": doc.get("model", "oracle") + "_" + config.run_name, # Either spanModel_relationModel or oracle_relationModel
            "needs": needs,
            "persons": persons,
            "relations": doc_relations,
            "tenure_ids": doc.get("tenure_ids", []),
            "household_members": doc.get("household_members", []),
        })
    config.logger.info(f"Generated predictions for {len(predictions)} documents")
    return predictions


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_model.py <path/to/final_model> [<data_path>]")
        sys.exit(1)

    config = Config(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else TEST_DATA)
    config.logger.info(f"Generating relation predictions (model={config.model_dir}, data={config.data_path.name})")

    # Load data
    data = load_json(config.data_path, config.logger)

    # Generate predictions
    predictions = predict_relations(data, config)
    model_names = predictions[0]['model']
    filename = config.output_filename.format(model_names=model_names)

    # Save predictions
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / filename
    save_json(output_path, predictions, config.logger)
    config.logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
