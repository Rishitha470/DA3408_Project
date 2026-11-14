"""Stage 2 - process: validate, clean and split the raw data with Ray Data.

Steps
 1. read data/raw/diabetes.csv into a Ray Dataset, split into blocks (parallel across CPUs)
 2. data-quality checks per block (nulls, out-of-range values, non-integer codes) -> aggregated report
 3. quality gate: fail the stage if too many rows are invalid or the class balance is implausible
 4. clean each block in parallel (drop invalid rows, cast types)
 5. stratified train / validation / test split -> parquet
 6. drift reference distribution computed from the training split
Run:  python -m src.data.process
"""
import json
import time
from typing import Dict

import numpy as np
import pandas as pd
import ray
from sklearn.model_selection import train_test_split

from src.common.config import DATA_PROCESSED, DATA_RAW, REPORTS_DIR, load_params
from src.common.drift import build_reference
from src.common.logging_utils import get_logger
from src.common.pipeline_log import track_stage
from src.common.schema import ALL_COLUMNS, FEATURES, TARGET

log = get_logger("process")
INTEGER_KINDS = {"binary", "ordinal"}


class DataQualityError(Exception):
    """Raised when the data fails the quality gate."""


def to_numeric(batch: pd.DataFrame) -> pd.DataFrame:
    """Coerce the schema columns to plain float64 (Ray may hand over Arrow-backed dtypes).
    Non-numeric values become NaN and are therefore counted as invalid."""
    out = pd.DataFrame(index=batch.index)
    for c in ALL_COLUMNS:
        if c in batch:
            out[c] = pd.to_numeric(batch[c].astype(object), errors="coerce").astype("float64")
        else:
            out[c] = np.nan
    return out


def invalid_mask(batch: pd.DataFrame) -> pd.Series:
    """True for rows that violate the schema (null, out of range, non-integer code, bad target)."""
    batch = to_numeric(batch)
    bad = batch[ALL_COLUMNS].isna().any(axis=1)
    for f in FEATURES:
        col = batch[f.name]
        bad |= (col < f.min) | (col > f.max)
        if f.kind in INTEGER_KINDS:
            bad |= (col.fillna(0) % 1) != 0
    bad |= ~batch[TARGET].isin([0, 1])
    return bad


def batch_quality(batch: pd.DataFrame) -> pd.DataFrame:
    """One-row summary of a block: row count, invalid rows, nulls and out-of-range per column."""
    batch = to_numeric(batch)
    stats: Dict[str, int] = {"rows": len(batch), "invalid_rows": int(invalid_mask(batch).sum()),
                             "positives": int((batch[TARGET] == 1).sum())}
    for f in FEATURES:
        col = batch[f.name]
        stats[f"null__{f.name}"] = int(col.isna().sum())
        stats[f"range__{f.name}"] = int(((col < f.min) | (col > f.max)).sum())
    stats[f"null__{TARGET}"] = int(batch[TARGET].isna().sum())
    return pd.DataFrame([stats])


def clean_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and cast to float (model input) / int (target)."""
    batch = to_numeric(batch)
    batch = batch[~invalid_mask(batch)].copy()
    batch[TARGET] = batch[TARGET].astype(int)
    return batch


def run_ray_pipeline(raw_path, num_cpus: int, num_blocks: int):
    """Returns (cleaned pandas DataFrame, quality summary dict, timing dict)."""
    timings = {}
    ray.init(num_cpus=num_cpus, include_dashboard=False, ignore_reinit_error=True,
             log_to_driver=False, configure_logging=False)
    try:
        t0 = time.perf_counter()
        ds = ray.data.read_csv(str(raw_path)).repartition(num_blocks).materialize()
        timings["read_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        quality = ds.map_batches(batch_quality, batch_format="pandas").to_pandas().sum().to_dict()
        timings["validate_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cleaned = ds.map_batches(clean_batch, batch_format="pandas").to_pandas()
        timings["clean_s"] = time.perf_counter() - t0
    finally:
        ray.shutdown()
    return cleaned, {k: int(v) for k, v in quality.items()}, timings


def quality_gate(q: Dict[str, int], cfg: Dict) -> Dict:
    """Evaluate the data-quality rules; returns the checks and raises DataQualityError on failure."""
    rows = max(q["rows"], 1)
    checks = {
        "min_rows": q["rows"] >= cfg["min_rows"],
        "max_invalid_fraction": q["invalid_rows"] / rows <= cfg["max_invalid_fraction"],
        "positive_rate_range": cfg["min_positive_rate"] <= q["positives"] / rows <= cfg["max_positive_rate"],
    }
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise DataQualityError(f"quality gate failed: {failed} (summary: rows={q['rows']}, "
                               f"invalid={q['invalid_rows']}, positives={q['positives']})")
    return checks


def main() -> None:
    params = load_params()
    cfg = params["process"]
    with track_stage("process") as rec:
        t_start = time.perf_counter()
        cleaned, quality, timings = run_ray_pipeline(DATA_RAW / "diabetes.csv",
                                                     cfg["num_cpus"], cfg["num_blocks"])
        checks = quality_gate(quality, cfg["quality"])
        duplicates = int(cleaned.duplicated().sum())
        if cfg["drop_duplicates"]:
            cleaned = cleaned.drop_duplicates()

        seed = params["seed"]
        train, rest = train_test_split(cleaned, test_size=cfg["val_size"] + cfg["test_size"],
                                       stratify=cleaned[TARGET], random_state=seed)
        rel_test = cfg["test_size"] / (cfg["val_size"] + cfg["test_size"])
        val, test = train_test_split(rest, test_size=rel_test, stratify=rest[TARGET], random_state=seed)

        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", train), ("val", val), ("test", test)):
            part.reset_index(drop=True).to_parquet(DATA_PROCESSED / f"{name}.parquet", index=False)
        (DATA_PROCESSED / "reference.json").write_text(json.dumps(build_reference(train)))

        total_s = time.perf_counter() - t_start
        rec.set_rows(quality["rows"], len(cleaned))
        report = {
            "rows_in": quality["rows"], "rows_out": len(cleaned),
            "invalid_rows_dropped": quality["invalid_rows"], "duplicate_rows": duplicates,
            "duplicates_dropped": bool(cfg["drop_duplicates"]),
            "positive_rate": round(float(cleaned[TARGET].mean()), 4),
            "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
            "nulls": {k[6:]: v for k, v in quality.items() if k.startswith("null__") and v},
            "out_of_range": {k[7:]: v for k, v in quality.items() if k.startswith("range__") and v},
            "quality_checks": checks,
            "ray": {"num_cpus": cfg["num_cpus"], "num_blocks": cfg["num_blocks"]},
            "timings_s": {k: round(v, 3) for k, v in timings.items()} | {"total_s": round(total_s, 3)},
            # end-to-end (includes Ray start-up) and processing-only throughput
            "throughput_rows_per_s": round(quality["rows"] / max(total_s, 1e-9), 1),
            "ray_processing_rows_per_s": round(quality["rows"] / max(sum(timings.values()), 1e-9), 1),
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
        log.info("processed %d -> %d rows in %.2fs (%.0f rows/s)", quality["rows"], len(cleaned),
                 total_s, report["throughput_rows_per_s"])


if __name__ == "__main__":
    np.random.seed(load_params()["seed"])
    main()
