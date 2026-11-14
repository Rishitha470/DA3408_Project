"""Stage 1 - ingest: download the CDC Diabetes Health Indicators dataset to data/raw/diabetes.csv.

Sources (params.yaml -> data.source):
  uci        UCI ML Repository dataset id 891 (253,680 rows) - default
  csv        an already downloaded CSV (e.g. Kaggle 'diabetes_binary_health_indicators_BRFSS2015.csv')
  synthetic  generated data with the same schema (offline smoke runs only)
Run:  python -m src.data.ingest
"""
import hashlib
import json

import pandas as pd

from src.common.config import DATA_RAW, REPORTS_DIR, ROOT, load_params
from src.common.logging_utils import get_logger
from src.common.pipeline_log import track_stage
from src.common.schema import ALL_COLUMNS, TARGET

log = get_logger("ingest")
UCI_DATASET_ID = 891
UCI_DIRECT_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"


def _from_uci() -> pd.DataFrame:
    """Fetch via the ucimlrepo client; fall back to the direct CSV link."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=UCI_DATASET_ID)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        log.info("downloaded %d rows via ucimlrepo", len(df))
        return df
    except Exception as exc:                              # network/API problems
        log.warning("ucimlrepo failed (%s); trying %s", exc, UCI_DIRECT_URL)
        return pd.read_csv(UCI_DIRECT_URL)


def load_source(params: dict) -> pd.DataFrame:
    source = params["data"]["source"]
    if source == "uci":
        df = _from_uci()
    elif source == "csv":
        df = pd.read_csv(ROOT / params["data"]["csv_path"])
    elif source == "synthetic":
        from src.data.synthetic import make_synthetic
        df = make_synthetic(params["data"]["synthetic_rows"], seed=params["seed"])
    else:
        raise ValueError(f"unknown data.source '{source}' (use uci, csv or synthetic)")
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")
    return df[ALL_COLUMNS]                                # drops extra columns such as ID


def main() -> None:
    params = load_params()
    with track_stage("ingest") as rec:
        df = load_source(params)
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        out = DATA_RAW / "diabetes.csv"
        df.to_csv(out, index=False)
        md5 = hashlib.md5(out.read_bytes()).hexdigest()
        rec.set_rows(len(df))
        rec["source"] = params["data"]["source"]
        info = {"source": params["data"]["source"], "rows": len(df), "columns": len(df.columns),
                "positive_rate": round(float(df[TARGET].mean()), 4), "md5": md5}
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "ingest.json").write_text(json.dumps(info, indent=2))
        log.info("wrote %s %s", out, info)


if __name__ == "__main__":
    main()
