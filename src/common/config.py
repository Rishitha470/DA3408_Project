"""Project paths and params.yaml loading."""
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"


def load_params(path: Path = ROOT / "params.yaml") -> Dict[str, Any]:
    """Load the DVC/MLflow parameter file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def tracking_uri() -> str:
    """MLflow tracking URI: env var wins, otherwise a local SQLite store in the repo root."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
