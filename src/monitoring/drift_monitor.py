"""Drift monitor: compares recent (encrypted) prediction inputs with the training distribution.

    python -m src.monitoring.drift_monitor            # compute and write reports/drift.json
    python -m src.monitoring.drift_monitor --retrain  # also run `dvc repro` when drift is detected

The backend computes the same PSI values continuously for Prometheus/Grafana; this CLI is the
batch job that decides on retraining (schedule it with cron, e.g. hourly).
"""
import argparse
import json
import subprocess
import sys

import pandas as pd

from src.common.config import LOGS_DIR, MODELS_DIR, REPORTS_DIR, ROOT, load_params
from src.common.drift import compute_drift
from src.common.logging_utils import get_logger
from src.common.pipeline_log import track_stage
from src.common.security import get_cipher, read_encrypted

log = get_logger("drift_monitor")
PREDICTION_LOG = LOGS_DIR / "predictions.enc"


def load_recent_inputs(last_n: int = 5000) -> pd.DataFrame:
    cipher = get_cipher()
    if cipher is None:
        raise RuntimeError("PREDICTION_LOG_KEY is not set; cannot decrypt the prediction log")
    rows = [r["inputs"] for r in read_encrypted(PREDICTION_LOG, cipher, last_n) if "inputs" in r]
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true", help="run `dvc repro` if drift is detected")
    ap.add_argument("--last-n", type=int, default=5000, help="number of recent predictions to use")
    args = ap.parse_args()
    params = load_params()["monitoring"]

    with track_stage("drift_check") as rec:
        reference = json.loads((MODELS_DIR / "drift_reference.json").read_text())
        current = load_recent_inputs(args.last_n)
        rec.set_rows(len(current))
        report = compute_drift(reference, current, params["drift_min_samples"])
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "drift.json").write_text(json.dumps(report, indent=2))
        rec["drift_detected"] = report["drift_detected"]
        log.info("drift status=%s detected=%s drifted=%s (n=%d)", report["status"],
                 report["drift_detected"], report["drifted_features"], report["n_samples"])

    if report["drift_detected"] and args.retrain and params["retrain_on_drift"]:
        # Re-run the whole DAG: re-ingest the latest data, re-process, retrain, re-evaluate.
        # The new model only becomes @champion if it passes the acceptance gate.
        log.warning("drift detected -> retraining with `dvc repro --force`")
        return subprocess.call(["dvc", "repro", "--force"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
