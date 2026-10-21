"""Logging helpers for the inference container.

Log files go to INFERENCE_LOG_DIR (default /opt/ml/processing/output/logs)
"""

import logging
import os
from pathlib import Path


def setup_logger(
    name: str,
    log_file_name: str | None = None,
    log_dir: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Reuse existing handlers if the logger is constructed more than once.
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file_name:
        target_dir = log_dir or Path(
            os.environ.get("INFERENCE_LOG_DIR", "/opt/ml/processing/output/logs")
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dir.chmod(
            0o700
        )  # User/owner can read/write/execute, no other users - fixes python:S5443
        file_handler = logging.FileHandler(target_dir / log_file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
