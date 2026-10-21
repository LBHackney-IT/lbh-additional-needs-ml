# Production Batch Inference Engine

This folder contains the production batch pipeline that extracts resident Additional Needs from unstructured text records (e.g., housing case notes, repairs descriptions) and maps predictions to database entity IDs.

## Pipeline Flow

```text
Athena (SQL) ──> Preprocessing ──> NER (Spans) ──> RE (Relations) ──> Entity Linking ──> Postprocessing ──> Partitioned JSON

```

1. **Preprocessing (`src/services/note_preprocessor.py`):** Groups note-tenure rows into a single record per note, applies category prefixes (`[Category: <category>]`), expands acronyms, and tags the `--source` identifier.
2. **NER / Span Extraction (`src/services/span_classifier_pipeline.py`):** Identifies text spans for needs (`need_category`) and people (`person_name`, `person_role`).
3. **RE / Relation Extraction (`src/services/relation_classifier_pipeline.py`):** Evaluates candidate (person, need) pairs within the note text to determine if they are related.
4. **EL / Entity Linking (`src/services/entity_linker.py`):** Applies deterministic rules to map predicted person mentions to household member IDs. Excludes minors (<18 years old) and defaults unresolved needs to tenure-level links.

---

## Folder Structure

| Functionality | Dir/File | Responsibility |
| --- | --- | --- |
| Composition root  | `src/run_inference.py` | Builds dependencies, configures the job, and runs batches |
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

### 1. Input Schema (Athena Query)

The input query expects an Athena table or view containing the following columns:

| Column | Type | Description |
| --- | --- | --- |
| `note_id` | `VARCHAR` | Unique identifier for the note |
| `note_category` | `VARCHAR` | Category string (prefixed to text during preprocessing) |
| `note_content` | `VARCHAR` | Raw unstructured text content |
| `note_date` | `TIMESTAMP` | Timestamp of note creation |
| `tenure_id` | `VARCHAR` | Linked tenancy identifier |
| `household_members` | `JSON Array` | Array of household member objects linked to the tenancy |

#### Default Filter (`--where`)

By default, queries apply MMH quality flags:

```sql
"flag_no_matching_target" = FALSE
AND "flag_note_date_parse_failed" = FALSE
AND "flag_is_organisation" = FALSE
AND "flag_inactive" = FALSE

```

To override this default (e.g., for date filtering or non-MMH datasets like Repairs), pass `--where` via CLI or `--where ""` to fetch all rows without filtering.

#### Preprocessor Schema Transformation

Upstream SQL returns one row per note-tenure match (possibly duplicated). `NotePreprocessor` groups these rows by `note_id`, consolidating multiple tenancies into a single pipeline object, expands acronyms (`LH` -> `Leaseholder`), and prefixes the category:

**Raw Athena Row:**

```json
{
  "note_id": "note-123",
  "note_category": "General",
  "note_content": "LH requires ground-floor accommodation.",
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

**Consolidated Preprocessed Record:**

```json
{
  "id": "note-123",
  "source": "mmh",
  "text": "[Category: General] Leaseholder requires ground-floor accommodation.",
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

*Ref: [Infrastructure Terraform](https://github.com/LBHackney-IT/Data-Platform/blob/main/terraform/core/71-housing-additional-needs-ml.tf) | [Upstram Airflow Reshape ETL*](https://github.com/LBHackney-IT/dap-airflow/blob/main/etl_scripts/housing/additional_needs/additional_needs_notes_reshape.py)

---

### 2. Output Schema (`/opt/ml/processing/output/part_*.json`)

Batch outputs are flushed periodically as partitioned JSON files containing enriched notes:

```json
{
  "id": "note-123",
  "source": "mmh",
  "needs": [
    {
      "id": "need-1",
      "text": "hoarding issue",
      "start": 187,
      "end": 285,
      "label": "housing_conditions_hoarding"
    }
  ],
  "persons": [
    {
      "id": "person-1",
      "text": "john smith",
      "start": 139,
      "end": 145,
      "label": "person_name"
    }
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
  "tenure_ids": [
    "b6468897-074d-4689-84ac-66d8a5891411"
  ],
  "household_members": [
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
  "model": "span-classifier-123-relation-extractor-2"
  "pipeline_run_at": "2026-10-07T16:00:00Z"
}

```

---

## Local Setup & Commands

All commands are run from the `/inference` directory.

### 1. Environment Setup & Tests

```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate
# Install dependencies
make install

# Run tests and linting
make test
make lint

```

### 2. Set Up Model Weights

Place weights in `./data/models/` (falls back to `0.5` threshold if `optimized_thresholds.json` is missing):

```text
data/models/
├── span_v1/
│   ├── config.json
│   └── final_model/ (model.safetensors, label_list.json, optimized_thresholds.json, tokenizer/)
└── relation_v1/
    ├── config.json
    └── final_model/ (model.safetensors, tokenizer/)

```

---

### 3. Execution Options

*Prerequisite:* AWS credentials with access to Housing Athena Workgroup.

For an AWS SSO profile, log in before running the pipeline:

```bash
aws sso login --profile data-platform-housing-staging
export AWS_PROFILE=data-platform-housing-staging
aws sts get-caller-identity
```

#### Option A: Direct Python Run

```bash
# Input/output dirs
mkdir -p data/models data/output
export AWS_PROFILE=data-platform-housing-staging

# Smoke test a single note
python -m src.run_inference \
  --model-root ./data/models \
  --output-dir ./data/output \
  --where "note_id = '<note-id>'"

# Target a custom dataset/table
python -m src.run_inference \
  --model-root ./data/models \
  --output-dir ./data/output \
  --athena-table v_repairs_jds_reshaped

```

#### Option B: Local Docker Container

```bash
# Build image
make build

# Single-note Docker smoke test
make smoke-test WHERE="note_id = '<note-id>'"

# Override the AWS profile and note filter
make run-docker \
  AWS_PROFILE=data-platform-housing-staging \
  ARGS='--where "note_id = '\''<note-id>'\''"'

# Custom table
make run-docker ARGS="--athena-table v_repairs_jds_reshaped"

```

#### Option C: Local SageMaker Job

`launch_sagemaker.py` is a local dev helper. It runs the published container through SageMaker Local Mode, reads models from
`s3://dataplatform-stg-housing-additional-needs-ml-storage/models/`, and writes predictions to `data/output/`. 

Your AWS profile needs access to the S3 bucket, Athena workgroup, and the ECR image.

```bash
pip install -e ".[launch]"

export SAGEMAKER_INFERENCE_IMAGE_URI="<account-id>.dkr.ecr.eu-west-2.amazonaws.com/housing-additional-needs-ml:<commit-sha>"
mkdir -p .sagemaker-local
export SAGEMAKER_LOCAL_HOME="$PWD/.sagemaker-local"

# Local smoke test
python launch_sagemaker.py \
  --where "note_id = '<note_id>'" \
  --span-model-name 'span-classifier-v1' \
  --relation-model-name 'relation-classifier-v1' \
  --profile 'data-platform-housing-staging' \
  --image-uri '<account-id>.dkr.ecr.eu-west-2.amazonaws.com/housing-additional-needs-ml:<commit-sha>'

```
This doesn't create a processing job in AWS but it's as close to production as you can get.