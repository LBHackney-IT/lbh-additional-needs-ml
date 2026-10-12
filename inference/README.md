# Production Inference

This folder contains the production batch application that extracts Additional Needs from housing case notes.

The application runs three stages:

```text
Athena records -> span extraction -> relation extraction -> entity linking -> partitioned output
```

- **NER (`src/pipelines/span_classifier_pipeline.py`):** Identifies text mentions (person_name, person_role, need_category)
- **RE (`src/pipelines/relation_classifier_pipeline.py`):** Pairs candidate need and person mentions within the same note.
- **EL (`src/pipelines/entity_linker.py`):** Deterministic business logic mapping person mentions to household member IDs (excludes minors <18; falls back to tenure-level links).   

## Folder Structure



| Functionality | Dir/File | Responsibility |
| --- | --- | --- |
| Composition root (`Program.cs` in a .NET project) | `src/run_inference.py` | Builds dependencies, configures the job, and runs batches |
| Infrastructure | `src/data/` | Reads production records through Athena |
| Domain definitions / DTOs | `src/common/schemas.py` | Defines types for spans, relations, links, household members, and notes |
| Application services / use cases | `src/pipelines/` | Coordinates span extraction, relation extraction, and linking |
| ML implementations | `src/classifiers/` | Contains custom transformer model components used by the pipelines |
| Unit and integration tests | `tests/` | Unit tests |

### Directory Map

```text
inference/
├── src/
│   ├── run_inference.py          # Composition root and batch orchestration
│   ├── common/                   # Schemas, logging, and JSON helpers
│   ├── data/                     # Athena record source and input shaping
│   ├── classifiers/              # Low-level model implementations
│   └── pipelines/                # Production model wrappers and business pipeline stages
├── tests/
│   ├── common/                   # Schema and contract tests
│   ├── data/                     # Athena adapter tests
│   ├── classifiers/              # Low-level model tests
│   ├── pipelines/                # Production pipeline behavior and rule tests
│   └── test_run_inference.py     # End-to-end orchestration tests with mocked stages
├── Dockerfile                    # Production container definition
├── pyproject.toml                # Package, dependency, pytest, and Ruff configuration
└── requirements.txt              # Production installation entry point
```

## Data Contracts

### Input Schema (Athena: `housing-refined-zone.additional_needs_notes_reshaped`)

Production data is read through Athena. The canonical record passed to the pipelines is shaped like this:

```json
{
  "id": "note-123",
  "text": "Mrs Smith requires ground-floor accommodation.",
  "tenure_ids": ["tenure-456"],
  "household_members": [
    {
      "id": "person-789",
      "fullName": "Mrs Smith",
      "dateOfBirth": "1980-01-01"
    }
  ]
}
```

See the [Data Platform repo](https://github.com/LBHackney-IT/Data-Platform/blob/main/terraform/core/71-housing-additional-needs-ml.tf) for the infrastructure and the [airflow repo](https://github.com/LBHackney-IT/dap-airflow/blob/main/etl_scripts/housing/additional_needs/additional_needs_notes_reshape.py) for the upstream ETL.

## Output Schema (`/opt/ml/processing/output/part_*.json`)

The application writes partitioned JSON files to its SageMaker Processing output directory.

```json
{
{
  "id": "note-123",
  "needs": [
     {
        "id": "need-1",
        "text": "hoarding issue",
        "start": 187,
        "end": 285,
        "label": "housing_conditions_hoarding"
    },
  ],
  "persons": [
    {
        "id": "person-1",
        "text": "john smith",
        "start": 139,
        "end": 145,
        "label": "person_name"
    },
  ],
  "relations": [
    {
      "from": "need-1",
      "to": "person-1"
    }
  ],
  "links": [
    {
      "need_id": "need-1",
      "target_id": "person-789",
      "target_type": "person",
      "confidence": 0.95,
      "method": "fuzzy_name"
    }
  ],
  "tenure_ids":[
    "b6468897-074d-4689-84ac-66d8a5891411"
  ],
  "household_members":[
    {
      "id": "person-789",
      "type": "person",
      "fullName": "John Smith",
      "isResponsible": true,
      "dateOfBirth": "1943-01-01 00:00:00",
      "personTenureType": "Tenant"
    }
  ],
  "pipeline_run_at": "2026-10-07T16:00:00Z"
}
}
```

## Local Setup and Tests

Run commands from this `/inference`, not from the repository root:

```bash
# Environment setup & tests
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src tests
```