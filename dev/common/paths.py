from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"

RAW = DATA / "raw"
ANNOTATIONS = RAW / "annotations"

PROCESSED = DATA / "processed"
MODELS = DATA / "models"

RESULTS = DATA / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"

LOGS = DATA / "logs"

# Data splits
TRAIN_DATA = PROCESSED / "train_data.json"
VAL_DATA = PROCESSED / "val_data.json"
TEST_DATA = PROCESSED / "test_data.json"

FULL_DATA = PROCESSED / "gold_standard.json"
