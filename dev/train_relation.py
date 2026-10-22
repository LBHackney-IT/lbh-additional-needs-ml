"""SageMaker entry point. Local mode only runs scripts from the root of source_dir."""

import runpy
from pathlib import Path

runpy.run_path(
    str(
        Path(__file__).resolve().parent
        / "relations"
        / "training"
        / "relation_extraction.py"
    ),
    run_name="__main__",
)
