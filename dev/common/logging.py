import logging

from transformers import TrainerCallback

from common.paths import LOGS


def setup_logger(name: str, log_file_name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when notebooks rerun cells
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console Settings
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file_name:
        # Log File Settings
        log_file = LOGS / log_file_name

        # Make sure it won't fail if the dir doesn't exist
        LOGS.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class FileLogCallback(TrainerCallback):
    """Forwards Trainer's per-step/per-epoch log dicts (loss, eval f1, etc.)
    into our own file logger, since HF's progress bar + internal logger
    don't write to it by default."""

    def __init__(self, logger, prefix=""):
        self.logger = logger
        self.prefix = prefix

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entries = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in logs.items()
        )
        self.logger.info("%sstep=%s %s", self.prefix, state.global_step, entries)
