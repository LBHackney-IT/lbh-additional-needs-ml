# Training & Experimentation

Contains preprocessing scripts and the logic for training and evaluating the NER and Relation Extraction models.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

_Training scripts assume you have access to the `data-platform-development` AWS account using the following profile name: `data-platform-admin-dev`._

### GPU Support

It is highly recommended to use a GPU-accelerated machine if you wish to train locally. Please ensure it has sufficient VRAM. Alternatively, this repo is set up to run training jobs using AWS Sagemaker GPU instances (`ml.g5.xlarge`) on the data platform dev account.

The remaining components, including data preprocessing, inference, and evaluation, can be run on a CPU-only machine.

Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for installation instructions for your platform.


## Running Scripts

### 1. Data Preprocessing

The preprocessing scripts prepare the dataset for annotation, training, and evaluation.

These are intended to run in order and are labelled numerically (`1_*.py`, `2_*.py`, etc.). Some may require AWS Access (see above).

The initial ETL to ingest and reshape MMH's notes data is stored in [DAP's Airflow repo](https://github.com/LBHackney-IT/dap-airflow/blob/main/etl_scripts/housing/additional_needs/additional_needs_notes_reshape.py).


### 2. Model Training

Commands in this section are run from `dev/` after installing that package.

There are two main models in this repo:

- span classification for Additional Needs & person entity extraction;
- relation classification for linking extracted needs to household members.

Training scripts are located under:

```bash
python spans/training/train_span.py
python relations/training/relation_extraction.py
```

Training configuration is defined using dataclasses at the top of each training script.

The span model also includes a separate threshold sweep script:

```bash
python spans/training/optimize_thresholds.py <path/to/final_model>
```

Training is computationally expensive and GPU acceleration is recommended.

### 3. Evaluation

Evaluation scripts run inference to generate model predictions and compare them against the annotated test set.

Evaluation is split into two stages:

1. **Prediction generation**
   Each extraction approach produces predictions in a standard format.
2. **Evaluation and comparison**
   Predictions are compared against the ground truth annotations using Precision/Recall/F1 (see below).

Evaluation does not require GPU acceleration (the span model takes ~5 minutes on CPU; the relation model ~10 minutes).

#### Span Extraction Evaluation

1. Generate predictions for each of the following models:
   * regex-based baseline: `spans/eval/predict_regex.py`
   * AWS Comprehend: `spans/eval/predict_comprehend.py`
   * custom span classifier: `spans/eval/predict_model.py ..data/./data/models/<MODEL_NAME>/final_model/`
   * gemini pre-annotations; (`utils/convert_gemini_annotations_to_predictions.py`)

2. Compare and evaluate
   `compare_eval_spans.py` loads the generated predictions and evaluates all approaches (configurable) against the test set and presents tables.

#### Relation Extraction Evaluation

1. Generate predictions for each of the following models:
   * relation extractor
   * closest match heuristic
   _You can choose an input file for the spans (NER step). This will default to the gold standard spans, but you can choose any model's outputs. This can measure cascading errors._
   ```bash
   python relations/eval/predict_*.py [model_path] [<data/results/predicted/file>]
   ```
2. Compare and evaluate
   `compare_eval_relations.py` loads the generated predictions and evaluates all configured approaches against the test set and presents tables.

#### Entity Linking

The entity linking heuristic script that writes CSV for the visualiser is:

```bash
python utils/match_needs_to_persons.py
```

It is not quantitatively evaluated due to a lack of annotated ground truth data. Results can be manually inspected using the visualiser (see below).

### 4. Visualisation

You can visually compare model outputs to the gold standard using the debugging UI. See [visualizer/README.md](../visualizer/README.md)