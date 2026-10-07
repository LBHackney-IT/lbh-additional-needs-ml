"""Logging helpers for the inference container.

Log files go to INFERENCE_LOG_DIR (default /tmp/inference-logs) so the
container does not depend on the R&D data/logs path layout.
"""

import logging
import os
from pathlib import Path


def setup_logger(name: str, log_file_name: str | None = None) -> logging.Logger:
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
        log_dir = Path(os.environ.get("INFERENCE_LOG_DIR", "/tmp/inference-logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
