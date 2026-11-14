"""Shared fixtures: a small synthetic dataset and a tiny trained model in a temp directory.

The API tests run against this model (INFERENCE_MODE=local), so they need no network,
no MLflow server and no Docker.
"""
import json
import os

import joblib
import numpy as np
import pytest
from cryptography.fernet import Fernet

from src.common.drift import build_reference
from src.common.schema import FEATURE_NAMES, TARGET
from src.data.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(3000, seed=7)


@pytest.fixture(scope="session")
def model_env(tmp_path_factory, synthetic_df):
    """Train a tiny LR model, write the files the backend expects, and point env vars at them."""
    from src.models.train import build_pipeline, tune_threshold

    root = tmp_path_factory.mktemp("serving")
    model_dir, reports_dir, log_dir = root / "models", root / "reports", root / "logs"
    for d in (model_dir, reports_dir, log_dir):
        d.mkdir()
    X, y = synthetic_df[FEATURE_NAMES], synthetic_df[TARGET].astype(int).to_numpy()
    pipe = build_pipeline("logistic_regression", {"C": 1.0}, 1.0, seed=0).fit(X, y)
    thr = tune_threshold(y, pipe.predict_proba(X)[:, 1], beta=2.0)
    meta = {"model_name": "test_lr", "model_version": "test_lr@0000", "threshold": thr,
            "run_id": "0" * 32, "features": FEATURE_NAMES}
    reference = build_reference(synthetic_df)
    joblib.dump({"pipeline": pipe, "threshold": thr, "model_version": meta["model_version"],
                 "meta": meta, "reference": reference}, model_dir / "model.joblib")
    (model_dir / "model_meta.json").write_text(json.dumps(meta))
    (model_dir / "drift_reference.json").write_text(json.dumps(reference))
    (reports_dir / "metrics.json").write_text(json.dumps({"roc_auc": 0.8, "recall": 0.75}))

    os.environ.update({
        "INFERENCE_MODE": "local", "MODEL_DIR": str(model_dir), "REPORTS_DIR": str(reports_dir),
        "LOG_DIR": str(log_dir), "PREDICTION_LOG_KEY": Fernet.generate_key().decode(),
        "ADMIN_TOKEN": "secret-token", "DRIFT_MIN_SAMPLES": "20", "DEPENDENCY_URLS": "",
    })
    return {"model_dir": model_dir, "reports_dir": reports_dir, "log_dir": log_dir, "threshold": thr}


@pytest.fixture
def patient():
    """A valid request body (higher-risk profile)."""
    return {"HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 33.0, "Smoker": 1, "Stroke": 0,
            "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
            "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4,
            "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 0, "Sex": 1, "Age": 10, "Education": 4,
            "Income": 4}


@pytest.fixture
def rng():
    return np.random.default_rng(0)
