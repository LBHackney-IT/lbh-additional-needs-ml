"""
Loads predictions from all methods and compares to gold standard.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tabulate import tabulate

from common.paths import VAL_DATA, TEST_DATA, METRICS, PREDICTIONS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from common.graph_helpers import plot_confusion_matrix
from shared.evaluators import SpanEvaluator, build_confusion_matrix

@dataclass
class Config:
    val_path: Path = TEST_DATA
    methods: List[str] = field(default_factory=lambda: ["regex", "comprehend", "span-classifier-4", "span-classifier-4-batched"]) # TODO: Make this into arg.

    def __post_init__(self):
        self.logger = setup_logger("eval.spans", "eval_spans.log")
        METRICS.mkdir(parents=True, exist_ok=True)
        self.results_path = METRICS / "span_comparison.json"
        self.summary_path = METRICS / "span_comparison_summary.csv"
        self.per_label_path = METRICS / "span_comparison_per_label.csv"


def load_method_predictions(method: str, config: Config) -> Dict:
    """Load predictions for a given method."""
    pred_path = PREDICTIONS / f"span.{method}.json"

    if not pred_path.exists():
        config.logger.warning("Predictions for %s not found at %s", method, pred_path)
        return {}

    predictions = load_json(pred_path, config.logger)
    # Index by record_id for easy lookup
    return {p["id"]: p for p in predictions}

def convert_preds_to_tuples(record):
    return [
        (n["start"], n["end"], n["label"])
        for n in record.get("needs", []) + record.get("persons", [])
        if "label" in n
    ]

def generate_summary_table(results_by_method, all_labels, output_path, logger):
    logger.info("Generating comparison summary...")
    summary_data = []
    for method, results in results_by_method.items():
        summary_data.append(
            {
                "Method": method,
                "Macro F1 (strict)": results.get("overall", {}).get("strict", {}).get("macro_f1", 0),
                "Micro F1 (strict)": results.get("overall", {}).get("strict", {}).get("micro_f1", 0),
                "Macro F1 (iou_0.5)": results.get("overall", {}).get("iou_0.5", {}).get("macro_f1", 0),
                "Micro F1 (iou_0.5)": results.get("overall", {}).get("iou_0.5", {}).get("micro_f1", 0),
                "Macro F1 (loose)": results.get("overall", {}).get("loose", {}).get("macro_f1", 0),
                "Micro F1 (loose)": results.get("overall", {}).get("loose", {}).get("micro_f1", 0),
            }
        )

    # Save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    logger.info("Summary saved to %s", output_path)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SPAN EVALUATION SUMMARY - ALL METHODS")
    logger.info("=" * 80)
    logger.info("\n"+tabulate(summary_df, headers="keys", showindex=False, tablefmt="psql", stralign="left", numalign="left", floatfmt=",.4f"))
    logger.info("=" * 80 + "\n")

def generate_per_label_table(results_by_method, all_labels, output_path, logger, match_strategy="iou_0.5", metric="f1"):
    """Build a per-label x method comparison table for a given match strategy."""
    rows = []
    for label in all_labels:
        row = {"Label": label}
        for method, results in results_by_method.items():
            try:
                value = results["per_label"][label][match_strategy][metric]
            except KeyError:
                value = 0.0
            row[method] = value
        rows.append(row)
    
    # Save
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    logger.info("Summary saved to %s", output_path)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PER-LABEL %s (%s) - ALL METHODS", metric.upper(), match_strategy.upper())
    logger.info("=" * 80)
    logger.info("\n"+tabulate(summary_df, headers="keys", showindex=False, tablefmt="psql", stralign="left", numalign="left", floatfmt=",.4f"))
    logger.info("=" * 80 + "\n")


def main():
    config = Config()
    config.logger.info(
        "Evaluating span predictions from %d methods: %s",
        len(config.methods),
        ", ".join(config.methods),
    )

    # Load ground truth (dict by id)
    val_records = load_json(config.val_path, config.logger)
    y_true_by_id = {
        r["id"]: convert_preds_to_tuples(r) for r in val_records
    }

    # Get all unique labels
    all_labels = sorted(
        set(label for spans in y_true_by_id.values() for _, _, label in spans)
    )
    config.logger.info("Found %d unique labels", len(all_labels))

    # Load predictions for each method
    all_predictions = {}
    for method in config.methods:
        all_predictions[method] = load_method_predictions(method, config)

    # Evaluate each method
    evaluator = SpanEvaluator(all_labels, config.logger)
    results_by_method = {}

    for method in config.methods:
        config.logger.info("Evaluating method: %s", method)
        predictions = all_predictions[method]

        # Populate y_true and y_pred (list of lists of tuples)
        y_true, y_pred = [], []

        for record_id, true_spans in y_true_by_id.items():
            y_true.append(true_spans)
            pred_spans = convert_preds_to_tuples(predictions.get(record_id, {}))
            y_pred.append(pred_spans)

        results = evaluator.evaluate(y_true, y_pred)
        results_by_method[method] = results
        evaluator.print_report(results, title=f"SPAN METRICS ({method.upper()})")

        cm = build_confusion_matrix(y_true, y_pred, all_labels, SpanEvaluator._score_loose)
        cm.to_csv(METRICS / f"confusion_{method}.csv")
        config.logger.info('Saved confusion matrix to %s.', (METRICS / f"confusion_{method}.csv"))
        plot_confusion_matrix(cm, f"Confusion Matrix — {method}", METRICS / f"confusion_{method}.png")


    # Save full results
    save_json(path=config.results_path, data=results_by_method, logger=config.logger)

    # Generate comparison tables
    generate_per_label_table(results_by_method, all_labels, config.per_label_path, config.logger)
    generate_summary_table(results_by_method, all_labels, config.summary_path, config.logger)

    config.logger.info(
        "Evaluation complete. Results saved to %s, %s, %s",
        config.results_path,
        config.per_label_path,
        config.summary_path
    )


if __name__ == "__main__":
    main()
