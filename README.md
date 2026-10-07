# Additional Needs Extraction (Machine Learning Platform)

This repository contains the source code for investigating the automatic extraction of **Additional Needs** from housing case notes.

The project applies Natural Language Processing (NLP) to identify references to additional needs (e.g. physical health, mental health, safeguarding concerns) within free-text housing records and, where possible, determine which household member each need refers to.

The system is implemented as a three-stage pipeline:

1. **Span extraction (NER)**, which identifies Additional Needs and person references within a case note.
2. **Relation extraction**, which links each identified need to the corresponding person reference.
3. **Entity linking**, which applies rule-based matching to resolve extracted needs to household member IDs.

---

## Repository Structure

This is a **monorepo**. Each directory owns its own dependencies.

| Folder | Purpose | Main Tech | README Link |
| :--- | :--- | :--- | :--- |
| **`inference/`** | Production container for batch jobs (Airflow / SageMaker) | PyTorch, Docker, pytest | [inference/README.md](./inference/README.md) |
| **`dev/`** | Model training, evaluation scripts, and dataset preprocessing. | PyTorch, Hugging Face, SageMaker | [dev/README.md](./dev/README.md) |
| **`visualizer/`** | Debugger UI for inspecting model predictions against ground truth | React, Vite, Node | [visualizer/README.md](./visualizer/README.md) |

Install and run each package from **inside that directory**.

_Note: If you update any model code in the `dev/` folder and want to sync those changes to production, you'll need to update `inference/`._
---

## Quick Start & AWS Environment

* **Python Version:** 3.12
* **Node Version:** v24+ (Visualizer only)
* **AWS Credentials:** Training and batch scripts require access to the `data-platform-development` account using profile `data-platform-admin-dev`.

To get started with a specific component, navigate into its folder and follow the readme:

```bash
# For production inference development
cd inference/

# For model training & experiments
cd dev/

# For running the visualizer
cd visualizer/
```

## Contributing

### Pre-commit Hooks

This repository uses [`pre-commit`](https://pre-commit.com/) to automatically strip output from Jupyter Notebooks before committing.

#### Setup

1. Install `pre-commit` (if not already installed):
   ```bash
   pip install pre-commit
   ```

2. Install the git hooks in your local clone:
   ```bash
   pre-commit install
   ```


#### Verification & Manual Run

To verify that the hook is installed correctly or to run `nbstripout` manually against all notebooks in the repository:

```bash
pre-commit run --all-files

```

If successful, you will see `nbstripout` pass (or modify staged notebooks to clear outputs) prior to committing.

### CI/CD

TODO
