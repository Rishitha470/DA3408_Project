"""Keeps recent prediction inputs (in memory + encrypted on disk) and computes live data drift."""
import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.common.drift import compute_drift
from src.common.logging_utils import get_logger
from src.common.security import append_encrypted, get_cipher, read_encrypted

log = get_logger("backend.store")


class PredictionStore:
    def __init__(self, log_dir: Path, window: int, min_samples: int):
        self.path = Path(log_dir) / "predictions.enc"
        self.recent = deque(maxlen=window)
        self.min_samples = min_samples
        self.cipher = get_cipher()
        self.lock = threading.Lock()
        self.reference: Optional[Dict] = None
        self.last_report: Dict = {"status": "not_computed", "drift_detected": False, "features": {}}
        if self.cipher is None:
            log.warning("PREDICTION_LOG_KEY not set: inputs are kept in memory only (not persisted)")

    def load_reference(self, model_dir: Path) -> None:
        path = Path(model_dir) / "drift_reference.json"
        self.reference = json.loads(path.read_text()) if path.exists() else None

    def warm_start(self) -> None:
        """Reload recent inputs from the encrypted log so drift survives restarts."""
        if self.cipher is None:
            return
        for rec in read_encrypted(self.path, self.cipher, self.recent.maxlen):
            if "inputs" in rec:
                self.recent.append(rec["inputs"])
        log.info("warm start: %d recent inputs loaded", len(self.recent))

    def record(self, inputs: Dict, output: Dict) -> None:
        with self.lock:
            self.recent.append(inputs)
        if self.cipher is not None:
            rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "inputs": inputs, "output": output}
            try:
                append_encrypted(self.path, rec, self.cipher)
            except OSError as exc:              # logging must never break a prediction
                log.error("could not write prediction log: %s", exc)

    def drift_report(self) -> Dict:
        if self.reference is None:
            self.last_report = {"status": "no_reference", "drift_detected": False, "features": {},
                                "n_samples": len(self.recent)}
            return self.last_report
        with self.lock:
            current = pd.DataFrame(list(self.recent))
        self.last_report = compute_drift(self.reference, current, self.min_samples)
        return self.last_report
