# Production Inference

This folder contains the production batch application that extracts Additional Needs from housing case notes.

The application runs four stages:

```text
Athena rows -> note preprocessing -> span extraction -> relation extraction -> entity linking -> partitioned output
```

- **Preprocessing (`src/services/note_preprocessor.py`):** Groups note-tenure rows into one record per note. Acronym expansion will run here too; the models were trained on expanded text.
- **NER (`src/services/span_classifier_pipeline.py`):** Identifies text mentions (person_name, person_role, need_category)
- **RE (`src/services/relation_classifier_pipeline.py`):** Pairs candidate need and person mentions within the same note.
- **EL (`src/services/entity_linker.py`):** Deterministic business logic mapping person mentions to household member IDs (excludes minors <18; falls back to tenure-level links).   

## Folder Structure



| Functionality | Dir/File | Responsibility |
| --- | --- | --- |
| Composition root (`Program.cs` in a .NET project) | `src/run_inference.py` | Builds dependencies, configures the job, and runs batches |
| Infrastructure gateway | `src/gateway/` | Reads production records through Athena |
| Domain definitions / DTOs | `src/common/schemas.py` | Defines types for spans, relations, links, household members, and notes |
| Application services | `src/services/` | Coordinates span extraction, relation extraction, and linking |
| ML implementations | `src/classifiers/` | Contains custom transformer model components used by the services |
| Unit and integration tests | `tests/` | Unit tests |

### Directory Map

```text
inference/
├── src/
│   ├── run_inference.py          # Composition root and batch orchestration
│   ├── common/                   # Schemas, logging, and JSON helpers
│   ├── gateway/                  # Athena record source
│   ├── classifiers/              # Low-level model implementations
│   └── services/                 # Preprocessing, model wrappers, and entity linking
├── tests/
│   ├── common/                   # Schema and contract tests
│   ├── gateway/                  # Athena adapter tests
│   ├── classifiers/              # Low-level model tests
│   ├── services/                 # Preprocessing, model wrapper, and linker tests
│   └── test_run_inference.py     # End-to-end orchestration tests with mocked stages
├── Dockerfile                    # Production container definition
├── pyproject.toml                # Package, dependency, pytest, and Ruff configuration
└── requirements.txt              # Production installation entry point
```

## Data Contracts

### Input Schema (Athena)

Production data is read through Athena. Any table can be selected at runtime, provided it has
the following columns:

- `note_id`
- `note_content`
- `note_date`
- `tenure_id`
- `household_members` (a JSON array)
- `flag_no_matching_target`
- `flag_note_date_parse_failed`
- `flag_is_organisation`
- `flag_inactive`

Each row is one note matched to one tenure. `household_members` is that tenure's household as a
flat JSON array, with role fields on the member (matches MMH tenure's `householdMembers` field):

```json
{
  "note_id": "note-123",
  "note_content": "Mrs Smith requires ground-floor accommodation.",
  "note_date": "2026-10-01 09:30:00",
  "tenure_id": "tenure-456",
  "household_members": [
    {
      "id": "person-789",
      "type": "person",
      "fullName": "Mrs Smith",
      "dateOfBirth": "1980-01-01",
      "personTenureType": "Tenant",
      "isResponsible": true
    }
  ]
}
```

A note matched to two tenancies is two rows. The gateway returns those rows after dropping any
where a flag is true. `NotePreprocessor` then groups them by `note_id` into this pipeline record:

```json
{
  "id": "note-123",
  "text": "Mrs Smith requires ground-floor accommodation.",
  "date": "2026-10-01 09:30:00",
  "tenure_ids": ["tenure-456"],
  "household_members": [
    {
      "id": "person-789",
      "type": "person",
      "fullName": "Mrs Smith",
      "dateOfBirth": "1980-01-01",
      "memberships": [
        {
          "tenure_id": "tenure-456",
          "personTenureType": "Tenant",
          "isResponsible": true
        }
      ]
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
      "dateOfBirth": "1943-01-01 00:00:00",
      "memberships": [
        {
          "tenure_id": "b6468897-074d-4689-84ac-66d8a5891411",
          "personTenureType": "Tenant",
          "isResponsible": true
        }
      ]
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

# Run locally (requires staging AWS credentials)
python -m src.run_inference \
  --athena-database housing-refined-zone \
  --athena-table v_repairs_jds_reshaped \
  --athena-workgroup housing \
  --output-dir ./data/output

# Docker Build & Run
docker build -t housing-additional-needs-ml .
docker run --rm \
  -v $(pwd)/data:/app/data \
  housing-additional-needs-ml \
  --athena-table v_repairs_jds_reshaped \
  --output-dir /app/data/output

# The table can alternatively be supplied as an environment variable.
docker run --rm \
  -e ATHENA_TABLE=v_repairs_jds_reshaped \
  housing-additional-needs-ml
```