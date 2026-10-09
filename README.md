# Additional Needs Extraction

This repository contains the code for extracting **Additional Needs** from free-text housing case notes and, where possible, attributing each need to the correct household member.

## Start Here

This is a monorepo with three separate parts:

| If you want to... | Go to... |
| --- | --- |
| Run the supported production pipeline | [`inference/`](./inference/), then read [`inference/README.md`](./inference/README.md) |
| Train models, prepare data, or compare experiments | [`dev/`](./dev/), then read [`dev/README.md`](./dev/README.md) |
| Inspect predictions manually in a browser | [`visualizer/`](./visualizer/), then read [`visualizer/README.md`](./visualizer/README.md) |
| Find source data, model artefacts, logs, or results | [`data/`](./data/) |

The most important distinction is this:

> `dev/` is where models and approaches are explored. `inference/` is the production runtime that must remain stable when a model is deployed.

## What the System Does

The production pipeline processes a case note in three stages:

1. **NER / span extraction** finds text spans that look like Additional Needs or people. This project uses _span classification_: the model scores candidate pieces of text rather than assigning a tag to every word.
2. **RE / relation extraction** examines each need/person pair and predicts whether that need refers to that person.
3. **EL / entity linking** uses deterministic business rules to map the predicted person mention to a real household member ID. If that cannot be done, the need is attributed to the relevant tenure instead.

In short: **find the mentions, connect the mentions, resolve the identity**.

![A three-stage entity attribution pipeline: NER, relation extraction, and entity linking are performed as sequential, independently trained stages](./docs/pipeline-overview.png)

The detailed input/output contract, examples, thresholds, fallback rules, and production troubleshooting guide live in [`inference/README.md`](./inference/README.md). This root README intentionally does not duplicate them.

---

## Repository Structure

This is a **monorepo**. Each directory owns its own dependencies and has its own README.

| Folder | Purpose | Main Tech | README Link |
| :--- | :--- | :--- | :--- |
| **`inference/`** | Production batch inference, model loading, validation, entity linking, and container configuration. | PyTorch, Docker, pytest | [inference/README.md](./inference/README.md) |
| **`dev/`** | Research code for preprocessing, training, evaluation, and experiments. | PyTorch, Hugging Face, SageMaker | [dev/README.md](./dev/README.md) |
| **`visualizer/`** | Local UI for inspecting predictions, annotations, relations, and attributions. | React, Vite, Node | [visualizer/README.md](./visualizer/README.md) |
| **`data/`** | Input data, processed datasets, model artefacts, logs, and experiment results. | CSV, JSON, spaCy, model files | — |

Install and run each package from **inside that directory**. The root directory is primarily for orientation; it is not a single installable Python package.

When a model or rule changes in `dev/`, production does not update automatically. A production update requires checking the packaged model, label list, thresholds, tokenizer, and production output schema in `inference/`.
---

## Environment and Quick Start

* **Python Version:** 3.12
* **Node Version:** v24+ (Visualizer only)
* **AWS credentials:** Training and some batch scripts require access to the `data-platform-development` account using profile `data-platform-admin-dev`.

To get started with a specific component, navigate into its folder and follow the readme:

```bash
# For production inference
cd inference/

# For model training & experiments
cd dev/

# For running the visualizer
cd visualizer/
```

## Pre-commit Hooks

This repository uses [`pre-commit`](https://pre-commit.com/) to automatically strip output from Jupyter Notebooks before committing.

### Setup

1. Install `pre-commit` (if not already installed):
   ```bash
   pip install pre-commit
   ```

2. Install the git hooks in your local clone:
   ```bash
   pre-commit install
   ```


### Verification & Manual Run

To verify that the hook is installed correctly or to run `nbstripout` manually against all notebooks in the repository:

```bash
pre-commit run --all-files
```

If successful, you will see `nbstripout` pass (or modify staged notebooks to clear outputs) prior to committing.