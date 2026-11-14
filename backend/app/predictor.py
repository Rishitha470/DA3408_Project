"""Inference engines behind a common interface (Strategy pattern).

MlflowServerPredictor - sends rows to the MLflow model server over REST (loose coupling; default in Docker)
LocalPredictor        - loads models/model.joblib in-process (development and unit tests)
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

from src.common.schema import FEATURE_NAMES


class PredictorError(Exception):
    """Inference failed or the model is unavailable (maps to HTTP 503)."""


class BasePredictor(ABC):
    """Common interface: returns the risk score for every row of a DataFrame of raw features."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.meta: Dict = {}

    def load_metadata(self) -> Dict:
        """Model card written by the training stage (name, version, threshold, metrics, ...)."""
        path = self.model_dir / "model_meta.json"
        if not path.exists():
            raise PredictorError(f"model metadata not found at {path}; run the training pipeline")
        self.meta = json.loads(path.read_text())
        return self.meta

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.5))

    @property
    def version(self) -> str:
        return self.meta.get("model_version", "unknown")

    @abstractmethod
    def load(self) -> None:
        """Prepare the predictor; raise PredictorError if it cannot serve."""

    @abstractmethod
    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Risk score in [0, 1] for each row."""


class LocalPredictor(BasePredictor):
    mode = "local"

    def load(self) -> None:
        import joblib
        self.load_metadata()
        path = self.model_dir / "model.joblib"
        try:
            bundle = joblib.load(path)
        except (OSError, EOFError, ValueError) as exc:
            raise PredictorError(f"cannot load model bundle {path}: {exc}") from exc
        self.pipeline = bundle["pipeline"]

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        try:
            return self.pipeline.predict_proba(rows[FEATURE_NAMES].astype(float))[:, 1]
        except Exception as exc:                       # any model failure becomes a 503
            raise PredictorError(f"local inference failed: {type(exc).__name__}") from exc


class MlflowServerPredictor(BasePredictor):
    mode = "mlflow"

    def __init__(self, model_dir: Path, url: str, timeout_s: float = 5.0):
        super().__init__(model_dir)
        self.url, self.timeout_s = url, timeout_s
        self.session = requests.Session()

    def load(self) -> None:
        self.load_metadata()

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        payload = {"dataframe_split": {"columns": FEATURE_NAMES,
                                       "data": rows[FEATURE_NAMES].astype(float).values.tolist()}}
        try:
            r = self.session.post(self.url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            preds = r.json()["predictions"]
            return np.array([float(p["probability"]) for p in preds])
        except requests.RequestException as exc:
            raise PredictorError(f"model server unreachable or returned an error: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictorError(f"unexpected model server response: {exc}") from exc


def make_predictor(settings) -> BasePredictor:
    """Factory selected by INFERENCE_MODE."""
    if settings.inference_mode == "mlflow":
        return MlflowServerPredictor(settings.model_dir, settings.model_server_url, settings.request_timeout_s)
    if settings.inference_mode == "local":
        return LocalPredictor(settings.model_dir)
    raise ValueError(f"INFERENCE_MODE must be 'mlflow' or 'local', got '{settings.inference_mode}'")
