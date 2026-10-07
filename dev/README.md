# Training and Experimentation

This folder is the **research and development workspace** for the Additional Needs pipeline. It contains preprocessing, model training, evaluation, baselines, and scripts for investigating new approaches.

It is not the production runtime. Production code lives in [`../inference/`](../inference/). A change here does not change production until a model or rule is deliberately promoted and checked against∂ production.

## The R&D Loop

Most work follows this sequence:

```text
prepare data -> train or run a baseline -> evaluate -> inspect errors -> decide whether to promote
```

The two learned components are:

- **Span model:** finds Additional Needs and person mentions in note text.
- **Relation model:** predicts which person mention a need refers to.

Entity linking is currently a rule-based step. The R&D linking script is useful for inspection, but it is not backed by a labelled evaluation set.

## Setup

Run these commands from `dev/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Training and SageMaker jobs require access to the `data-platform-development` AWS account with the `data-platform-admin-dev` profile. Training is intended for a GPU machine or the configured SageMaker `ml.g5.xlarge` job; preprocessing and evaluation can generally run on CPU.

## Workflow

### 1. Prepare data

The scripts in [`preprocessing/`](./preprocessing/) prepare data for annotation, training, and evaluation. Numbered scripts are intended to be followed in order. Read the script configuration before running one because some steps use AWS or write to shared data locations.

The upstream ETL that ingests and reshapes MMH notes is maintained in [DAP's Airflow repository](https://github.com/LBHackney-IT/dap-airflow/blob/main/etl_scripts/housing/additional_needs/additional_needs_notes_reshape.py).

### 2. Train models

Training configuration is defined near the top of each training script:

```bash
python spans/training/train_span.py
python relations/training/relation_extraction.py
```

The span model has a separate threshold optimisation step. Run it against the completed span model directory:

```bash
python spans/training/optimize_thresholds.py <path/to/final_model>
```

For SageMaker training, review the settings in [`launch_sagemaker.py`](./launch_sagemaker.py), especially the AWS profile, S3 paths, role, and instance type, before starting a job.

### 3. Evaluate predictions

Evaluation has two separate questions:

1. **How well does a method find the right spans or relations?** Compare predictions with annotated ground truth using precision, recall, and F1.
2. **Where does the pipeline fail?** Use the visualizer to inspect examples and understand missed spans, incorrect relations, and linking errors.

Generate span predictions with the scripts in [`spans/eval/`](./spans/eval/), then compare them with:

```bash
python spans/eval/compare_eval_spans.py
```

Available span approaches include the regex baseline, AWS Comprehend, the custom span model, and converted Gemini annotations. The exact input and output locations are defined in each script's configuration.

Generate relation predictions with the scripts in [`relations/eval/`](./relations/eval/):

```bash
python relations/eval/predict_model.py <path/to/model> [path/to/span_predictions.json]
python relations/eval/compare_eval_relations.py
```

The optional span prediction file lets you measure cascading errors: use gold spans to evaluate relation extraction in isolation, or predicted spans to measure end-to-end behavior.

### 4. Inspect entity linking

The heuristic linker can write inspection data for the visualizer:

```bash
python utils/match_needs_to_persons.py <path/to/data.json>
```

Entity linking is not currently evaluated with precision/recall because there is no labelled ground-truth linking dataset. Treat visual inspection as diagnostic evidence, not as a directly comparable model score.

### 5. View results

Use the [visualizer](../visualizer/README.md) to compare predictions with annotations and inspect relation and attribution behavior.

## Promoting a Model

Before using a model in production, check more than its headline F1 score:

- label names and label ordering match the production model package;
- confidence thresholds have been generated and copied with the model;
- tokenizer and maximum sequence length are compatible;
- predictions use the production span/relation schema;
- representative errors have been reviewed in the visualizer;
- production inference tests pass after the model is packaged.

The dissertation contains the full research rationale and experiment history. This README is deliberately only the runbook: what to run, what it produces, and what must be checked before promotion.