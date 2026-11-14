"""Backend settings, all configurable through environment variables (12-factor style)."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _parse_pairs(raw: str) -> Dict[str, str]:
    """'a=http://x,b=http://y' -> {'a': 'http://x', 'b': 'http://y'}"""
    out = {}
    for item in filter(None, (p.strip() for p in raw.split(","))):
        name, _, url = item.partition("=")
        if name and url:
            out[name.strip()] = url.strip()
    return out


@dataclass
class Settings:
    # "mlflow": call the MLflow model server over REST (default in Docker)
    # "local" : load models/model.joblib in-process (development, tests)
    inference_mode: str = field(default_factory=lambda: os.getenv("INFERENCE_MODE", "local"))
    model_server_url: str = field(default_factory=lambda: os.getenv(
        "MODEL_SERVER_URL", "http://localhost:5001/invocations"))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", ROOT / "models")))
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", ROOT / "reports")))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", ROOT / "logs")))
    dvc_file: Path = field(default_factory=lambda: Path(os.getenv("DVC_FILE", ROOT / "dvc.yaml")))
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_S", "5")))
    cors_origins: List[str] = field(default_factory=lambda: [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()])
    refresh_interval_s: float = field(default_factory=lambda: float(os.getenv("REFRESH_INTERVAL_S", "15")))
    # drift is computed on the most recent N inputs; a small window reacts to a shift quickly instead of
    # diluting it with older traffic (keep equal to monitoring.drift_window in params.yaml)
    drift_window: int = field(default_factory=lambda: int(os.getenv("DRIFT_WINDOW", "500")))
    drift_min_samples: int = field(default_factory=lambda: int(os.getenv("DRIFT_MIN_SAMPLES", "100")))
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    # internal health checks of other services, e.g. "model-server=http://model-server:5001/ping"
    dependencies: Dict[str, str] = field(default_factory=lambda: _parse_pairs(os.getenv("DEPENDENCY_URLS", "")))
    # browser-facing links shown in the pipeline console
    tool_links: Dict[str, str] = field(default_factory=lambda: _parse_pairs(os.getenv(
        "TOOL_LINKS",
        "MLflow=http://localhost:5000,Grafana=http://localhost:3000,"
        "Prometheus=http://localhost:9090,Alertmanager=http://localhost:9093")))


settings = Settings()
