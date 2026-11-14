"""Prometheus instrumentation of the backend, the model, the data and the training pipeline."""
import calendar
import json
import time
from pathlib import Path
from typing import Dict, List

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------- API (RED metrics)
HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "endpoint", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"],
                         buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
HTTP_IN_PROGRESS = Gauge("http_requests_in_progress", "Requests currently being processed")
APP_EXCEPTIONS = Counter("app_exceptions_total", "Handled exceptions by type", ["type"])

# ---------------------------------------------------------------- model
PREDICTIONS = Counter("model_predictions_total", "Predictions by risk level", ["risk_level"])
PREDICTION_SCORE = Histogram("model_prediction_probability", "Distribution of risk scores",
                             buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
INFERENCE_LATENCY = Histogram("model_inference_duration_seconds", "Model inference latency (incl. explanation)",
                              buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
MODEL_INFO = Gauge("model_info", "Currently served model (value 1)", ["model_name", "model_version", "inference_mode"])
MODEL_TEST_METRIC = Gauge("model_test_metric", "Test-set metric of the served model", ["metric"])
MODEL_THRESHOLD = Gauge("model_decision_threshold", "Decision threshold of the served model")

# ---------------------------------------------------------------- data drift
DRIFT_PSI = Gauge("data_drift_psi", "Population Stability Index per feature", ["feature"])
DRIFT_DETECTED = Gauge("data_drift_detected", "1 if any feature has PSI >= 0.2")
DRIFT_SAMPLES = Gauge("data_drift_window_samples", "Recent inputs used for drift computation")

# ---------------------------------------------------------------- pipeline / data engineering
PIPE_STATUS = Gauge("pipeline_stage_last_status", "Last run of the stage: 1 success, 0 failed", ["stage"])
PIPE_DURATION = Gauge("pipeline_stage_last_duration_seconds", "Duration of the last run", ["stage"])
PIPE_THROUGHPUT = Gauge("pipeline_stage_rows_per_second", "Rows/s of the last run", ["stage"])
PIPE_LAST_RUN = Gauge("pipeline_stage_last_run_timestamp_seconds", "Unix time of the last run", ["stage"])
PIPE_RUNS = Gauge("pipeline_stage_runs", "Recorded runs per stage and status", ["stage", "status"])
DATA_ROWS = Gauge("data_pipeline_rows", "Rows in / out of the Ray processing stage", ["kind"])
DATA_RAY_THROUGHPUT = Gauge("data_pipeline_ray_rows_per_second", "Ray processing throughput (excl. start-up)")

# ---------------------------------------------------------------- other services
DEPENDENCY_UP = Gauge("dependency_up", "1 if the service answered its health check", ["service"])


def _read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def update_model_gauges(meta: Dict, mode: str, reports_dir: Path) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(meta.get("model_name", "unknown"), meta.get("model_version", "unknown"), mode).set(1)
    MODEL_THRESHOLD.set(float(meta.get("threshold", 0.5)))
    test = _read_json(reports_dir / "metrics.json")
    for k in ("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"):
        if k in test:
            MODEL_TEST_METRIC.labels(k).set(float(test[k]))


def update_pipeline_gauges(runs: List[Dict], reports_dir: Path) -> None:
    """runs: newest first (see src.common.pipeline_log.read_runs)."""
    seen, counts = set(), {}
    for r in runs:
        stage, status = r.get("stage", "?"), r.get("status", "?")
        counts[(stage, status)] = counts.get((stage, status), 0) + 1
        if stage in seen:
            continue
        seen.add(stage)
        PIPE_STATUS.labels(stage).set(1 if status == "success" else 0)
        PIPE_DURATION.labels(stage).set(float(r.get("duration_s", 0)))
        PIPE_THROUGHPUT.labels(stage).set(float(r.get("rows_per_s", 0)))
        try:
            ts = calendar.timegm(time.strptime(r["finished_at"][:19], "%Y-%m-%dT%H:%M:%S"))  # UTC
            PIPE_LAST_RUN.labels(stage).set(ts)
        except (KeyError, ValueError):
            pass
    for (stage, status), n in counts.items():
        PIPE_RUNS.labels(stage, status).set(n)
    dq = _read_json(reports_dir / "data_quality.json")
    if dq:
        DATA_ROWS.labels("in").set(dq.get("rows_in", 0))
        DATA_ROWS.labels("out").set(dq.get("rows_out", 0))
        DATA_RAY_THROUGHPUT.set(dq.get("ray_processing_rows_per_s", 0))


def update_drift_gauges(report: Dict) -> None:
    DRIFT_SAMPLES.set(report.get("n_samples", 0))
    DRIFT_DETECTED.set(1 if report.get("drift_detected") else 0)
    for name, v in report.get("features", {}).items():
        DRIFT_PSI.labels(name).set(v["psi"])
