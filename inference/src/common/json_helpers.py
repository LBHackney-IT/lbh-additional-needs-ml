"""JSON load / save helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def save_json(path: Path, data: Any, logger: logging.Logger | None = None) -> None:
    if logger:
        logger.info("Saving data to %s...", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_json(path: Path, logger: logging.Logger | None = None) -> Any:
    if logger:
        logger.info("Loading data from %s...", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)
