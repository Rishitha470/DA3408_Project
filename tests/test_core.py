"""Unit tests: schema, feature engineering, encryption, pipeline run log, DVC DAG parsing."""
import json

import pandas as pd
import pytest
from cryptography.fernet import Fernet

from backend.app.pipeline_status import load_dag
from src.common.config import ROOT
from src.common.features import ENGINEERED, add_features
from src.common.pipeline_log import read_runs, track_stage
from src.common.schema import FEATURE_NAMES, FEATURES, risk_level
from src.common.security import append_encrypted, get_cipher, read_encrypted


# ---------------------------------------------------------------- schema
def test_schema_has_21_unique_features():
    assert len(FEATURE_NAMES) == 21 and len(set(FEATURE_NAMES)) == 21


def test_healthy_reference_values_are_within_range():
    for f in FEATURES:
        if f.healthy is not None:
            assert f.min <= f.healthy <= f.max, f.name


@pytest.mark.parametrize("p, expected", [(0.0, "low"), (0.29, "low"), (0.30, "moderate"),
                                         (0.49, "moderate"), (0.50, "high"), (0.99, "high")])
def test_risk_level_bands_follow_threshold(p, expected):
    assert risk_level(p, threshold=0.5) == expected       # moderate band starts at 0.6 * 0.5


# ---------------------------------------------------------------- features
def test_add_features_appends_engineered_columns(synthetic_df):
    out = add_features(synthetic_df[FEATURE_NAMES].head(50))
    assert list(out.columns) == FEATURE_NAMES + ENGINEERED
    assert len(out) == 50


def test_feature_values_are_correct():
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    row.update(BMI=31, HighBP=1, HighChol=1, PhysActivity=1, Fruits=1, Smoker=1, MentHlth=20, PhysHlth=25)
    out = add_features(pd.DataFrame([row])).iloc[0]
    assert out["bmi_category"] == 3            # obese
    assert out["cardio_risk_count"] == 2
    assert out["lifestyle_score"] == 1         # 1 + 1 + 0 - 1 - 0
    assert out["poor_health_days"] == 30       # capped


def test_add_features_does_not_modify_input(synthetic_df):
    before = synthetic_df.copy()
    add_features(synthetic_df[FEATURE_NAMES])
    pd.testing.assert_frame_equal(before, synthetic_df)


# ---------------------------------------------------------------- encryption
def test_encrypted_log_roundtrip(tmp_path):
    cipher = Fernet(Fernet.generate_key())
    path = tmp_path / "p.enc"
    for i in range(3):
        append_encrypted(path, {"i": i, "BMI": 30}, cipher)
    assert [r["i"] for r in read_encrypted(path, cipher)] == [0, 1, 2]
    assert b"BMI" not in path.read_bytes()                       # really encrypted on disk


def test_wrong_key_cannot_read_log(tmp_path):
    path = tmp_path / "p.enc"
    append_encrypted(path, {"secret": 1}, Fernet(Fernet.generate_key()))
    assert list(read_encrypted(path, Fernet(Fernet.generate_key()))) == []


def test_no_key_means_no_cipher(monkeypatch):
    monkeypatch.delenv("PREDICTION_LOG_KEY", raising=False)
    assert get_cipher() is None


# ---------------------------------------------------------------- pipeline run log
def test_track_stage_records_success(tmp_path):
    runs = tmp_path / "runs.jsonl"
    with track_stage("demo", runs) as rec:
        rec.set_rows(100, 90)
    r = read_runs(runs)[0]
    assert r["stage"] == "demo" and r["status"] == "success"
    assert r["rows_out"] == 90 and r["duration_s"] >= 0


def test_track_stage_records_failure_and_reraises(tmp_path):
    runs = tmp_path / "runs.jsonl"
    with pytest.raises(ValueError):
        with track_stage("demo", runs):
            raise ValueError("boom")
    r = read_runs(runs)[0]
    assert r["status"] == "failed" and "boom" in r["error"]


def test_read_runs_newest_first_and_skips_corrupt_lines(tmp_path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps({"stage": "a"}) + "\nnot json\n" + json.dumps({"stage": "b"}) + "\n")
    assert [r["stage"] for r in read_runs(runs)] == ["b", "a"]


# ---------------------------------------------------------------- DVC DAG
def test_dvc_dag_edges_match_pipeline():
    dag = load_dag(ROOT / "dvc.yaml")
    assert [s["name"] for s in dag["stages"]] == ["ingest", "process", "train", "evaluate"]
    for edge in (["ingest", "process"], ["process", "train"], ["train", "evaluate"]):
        assert edge in dag["edges"]


def test_dag_falls_back_when_dvc_file_missing(tmp_path):
    dag = load_dag(tmp_path / "missing.yaml")
    assert len(dag["stages"]) == 4 and ["train", "evaluate"] in dag["edges"]
