"""Unit and integration tests of the data pipeline (validation, cleaning, quality gate, Ray)."""
import numpy as np
import pandas as pd
import pytest

from src.common.schema import ALL_COLUMNS, TARGET
from src.data.ingest import load_source
from src.data.process import (DataQualityError, batch_quality, clean_batch, invalid_mask,
                              quality_gate, run_ray_pipeline, split_data)

GATE = {"min_rows": 10, "max_invalid_fraction": 0.05, "min_positive_rate": 0.02, "max_positive_rate": 0.5}


def corrupt(df):
    bad = df.head(6).copy()
    bad.loc[bad.index[0], "BMI"] = np.nan          # null
    bad.loc[bad.index[1], "Age"] = 14              # out of range
    bad.loc[bad.index[2], "GenHlth"] = 2.5         # non-integer code
    bad.loc[bad.index[3], TARGET] = 2              # invalid label
    return bad


def test_invalid_mask_flags_each_kind_of_problem(synthetic_df):
    mask = invalid_mask(corrupt(synthetic_df))
    assert mask.tolist()[:4] == [True, True, True, True]
    assert not mask.iloc[4:].any()


def test_invalid_mask_accepts_arrow_dtypes(synthetic_df):
    """Ray hands over Arrow-backed columns; validation must still work."""
    arrow = synthetic_df.head(20).astype("float64[pyarrow]")
    assert not invalid_mask(arrow).any()


def test_non_numeric_values_are_invalid(synthetic_df):
    df = synthetic_df.head(3).astype(object)
    df.loc[df.index[0], "Income"] = "abc"
    assert invalid_mask(df).tolist() == [True, False, False]


def test_batch_quality_counts(synthetic_df):
    q = batch_quality(corrupt(synthetic_df)).iloc[0]
    assert q["rows"] == 6 and q["invalid_rows"] == 4
    assert q["null__BMI"] == 1 and q["range__Age"] == 1


def test_clean_batch_drops_invalid_rows_and_casts(synthetic_df):
    out = clean_batch(corrupt(synthetic_df))
    assert len(out) == 2
    assert out[TARGET].dtype.kind == "i"


@pytest.mark.parametrize("q, ok", [
    ({"rows": 100, "invalid_rows": 2, "positives": 10}, True),
    ({"rows": 100, "invalid_rows": 6, "positives": 10}, False),    # too many invalid rows
    ({"rows": 5, "invalid_rows": 0, "positives": 1}, False),       # too few rows
    ({"rows": 100, "invalid_rows": 0, "positives": 1}, False),     # implausible class balance
])
def test_quality_gate(q, ok):
    if ok:
        assert all(quality_gate(q, GATE).values())
    else:
        with pytest.raises(DataQualityError):
            quality_gate(q, GATE)


def test_split_is_reproducible_regardless_of_row_order(synthetic_df):
    """Reproducibility: Ray may return rows in a different order on each run; the seeded split
    must still select exactly the same rows for train, validation and test."""
    data = clean_batch(synthetic_df.head(2000))
    shuffled = data.sample(frac=1.0, random_state=7)
    first = split_data(data, 0.15, 0.15, seed=42)
    second = split_data(shuffled, 0.15, 0.15, seed=42)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    train, val, test = first
    assert len(train) + len(val) + len(test) == len(data)


def test_ingest_synthetic_source_has_schema():
    df = load_source({"seed": 1, "data": {"source": "synthetic", "synthetic_rows": 200}})
    assert list(df.columns) == ALL_COLUMNS and len(df) == 200


def test_ingest_rejects_unknown_source():
    with pytest.raises(ValueError):
        load_source({"seed": 1, "data": {"source": "nope"}})


def test_ray_pipeline_end_to_end(tmp_path, synthetic_df):
    """Integration: Ray reads, validates and cleans a CSV containing bad rows."""
    raw = pd.concat([synthetic_df.head(500), corrupt(synthetic_df.tail(10))])
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    cleaned, quality, timings = run_ray_pipeline(path, num_cpus=1, num_blocks=2)
    assert quality["rows"] == 506 and quality["invalid_rows"] == 4
    assert len(cleaned) == 502
    assert set(timings) == {"read_s", "validate_s", "clean_s"}
