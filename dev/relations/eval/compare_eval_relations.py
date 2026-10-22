"""
Loads predictions from all methods and compares to gold standard.
Handles both 'oracle' classifiers (ones with gold standard spans fed in) and cascaded classifiers
by matching predicted spans to gs spans using IOU matching.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from common.data_utils import span_iou
from common.json_helpers import load_json, save_json
from common.logging import setup_logger
from common.paths import METRICS, PREDICTIONS, TEST_DATA
from shared.evaluators import RelationEvaluator
from tabulate import tabulate


@dataclass
class Config:
    data_path: Path = TEST_DATA
    # Methods: Uncomment depending on whether comparing oracle or cascading models
    methods: list[str] = field(
        default_factory=lambda: [
            "gemini",
            "oracle_closest_match",
            "oracle_relation-classifier-2",
            "span-classifier-4_closest_match",
            "span-classifier-4_relation-classifier-2",
        ]
    )  # TODO: Make this into arg.

    def __post_init__(self):
        self.logger = setup_logger(
            "eval.relations",
            "eval_relations.log",
        )
        self.results_path = METRICS / "relations_comparison.json"
        self.summary_path = METRICS / "relations_comparison_summary.csv"


def build_id_map(
    pred_entities: list[dict], gold_entities: list[dict], iou_threshold: float = 0.5
) -> dict[str, str]:
    """Map predicted annotation entity ids onto gold entity ids using IoU matching."""
    candidates = []

    for pred in pred_entities:
        for gold in gold_entities:
            if pred["label"] != gold["label"]:
                continue

            iou = span_iou(pred["start"], pred["end"], gold["start"], gold["end"])
            if iou > 0:
                candidates.append((iou, pred["id"], gold["id"]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_pred = set()
    used_gold = set()
    id_map = {}

    for iou, pred_id, gold_id in candidates:
        if pred_id in used_pred or gold_id in used_gold:
            continue
        if iou < iou_threshold:
            continue

        id_map[pred_id] = gold_id
        used_pred.add(pred_id)
        used_gold.add(gold_id)

    return id_map


def load_method_predictions(method: str, gold_data, logger) -> dict:
    """Load predictions for a given method and map spans onto gold standard (expects full record list)"""
    pred_path = PREDICTIONS / f"relation.{method}.json"

    if not pred_path.exists():
        logger.warning("Predictions for %s not found at %s", method, pred_path)
        return None

    # Load raw predictions
    predictions = load_json(pred_path, logger)
    gold_records_by_id = {r["id"]: r for r in gold_data}

    # Map spans onto GS spans
    result = {}
    for record in predictions:
        record_id = record["id"]
        gold_record = gold_records_by_id[record_id]
        # ID Map: pred_id -> gs_id
        id_map = build_id_map(
            record.get("needs", []) + record.get("persons", []),
            gold_record.get("needs", []) + gold_record.get("persons", []),
        )

        # Translate relation annotations onto the real gold entity ids. (if oracle, will just map ids to themselves)
        mapped_relations = []
        for r in record["relations"]:
            # Get mapped span IDs
            mapped_from = id_map.get(r["from"])
            mapped_to = id_map.get(r["to"])
            if not mapped_from or not mapped_to:
                continue
            # Add to relations list (list of tuples)
            mapped_relations.append((mapped_from, mapped_to))

        # Save as dict: id: relations list of tuples
        result[record_id] = mapped_relations

    return result


def generate_comparison_table(results_by_method, summary_path, logger):
    logger.info("Generating comparison summary...")
    summary_data = [
        {
            "Method": method,
            "F1": results.get("f1"),
            "Precision": results.get("precision"),
            "Recall": results.get("recall"),
            "TP": results.get("tp"),
            "FP": results.get("fp"),
            "FN": results.get("fn"),
        }
        for method, results in results_by_method.items()
    ]

    # Save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved to %s", summary_path)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("RELATION EXTRACTION METRICS")
    logger.info("=" * 80)
    logger.info(
        "\n"
        + tabulate(
            summary_df,
            headers="keys",
            showindex=False,
            tablefmt="psql",
            stralign="left",
            numalign="left",
            floatfmt=",.4f",
        )
    )
    logger.info("=" * 80 + "\n")


def main():
    config = Config()
    config.logger.info(
        "Evaluating relations predictions from %d methods: %s",
        len(config.methods),
        ", ".join(config.methods),
    )

    # Load gold standard
    val_records = load_json(config.data_path, config.logger)
    y_true_by_id = {
        record["id"]: [(r["from"], r["to"]) for r in record.get("relations", [])]
        for record in val_records
    }

    # Load predictions for each method
    all_predictions = {}
    for method in config.methods:
        pred = load_method_predictions(method, val_records, config.logger)
        if pred is not None:
            all_predictions[method] = pred

    # Evaluate each method
    evaluator = RelationEvaluator(config.logger)
    results_by_method = {}
    for method, predictions in all_predictions.items():
        config.logger.info("Evaluating method: %s", method)

        # Build y_pred and y_true (lists of lists of (from, to) tuples)
        y_true = []
        y_pred = []
        for record_id, true_relations in y_true_by_id.items():
            y_true.append(true_relations)
            y_pred.append(predictions.get(record_id, []))

        # Evaluate
        results = evaluator.evaluate(y_pred, y_true)
        results_by_method[method] = results

    # Save full results
    save_json(config.results_path, results_by_method, config.logger)

    generate_comparison_table(results_by_method, config.summary_path, config.logger)
    config.logger.info(
        "Evaluation complete. Results saved to %s and %s",
        config.results_path,
        config.summary_path,
    )


if __name__ == "__main__":
    main()
