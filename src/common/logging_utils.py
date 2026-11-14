"""Consistent logging to console and to logs/<name>.log."""
import logging
import os
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Return a logger that writes to stderr and to <log_dir>/<name>.log (created on demand)."""
    logger = logging.getLogger(name)
    if logger.handlers:                       # already configured
        return logger
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(console)
    log_dir = log_dir or os.environ.get("LOG_DIR") or str(Path(__file__).resolve().parents[2] / "logs")
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name.split('.')[-1]}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(fh)
    except OSError as exc:                    # read-only filesystem: console logging only
        logger.warning("file logging disabled: %s", exc)
    logger.propagate = False
    return logger
