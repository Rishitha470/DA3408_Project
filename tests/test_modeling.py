"""Unit tests of training, evaluation, drift and explanation logic."""
import numpy as np
import pytest

from backend.app.explain import top_factors, what_if_rows
from src.common.drift import build_reference, compute_drift, psi
from src.common.schema import FEATURE_NAMES, TARGET
from src.models.evaluate import check_acceptance
from src.models.train import (build_pipeline, classification_metrics, feature_importance,
                              tune_threshold)


@pytest.fixture(scope="module")
def xy(synthetic_df):
    return synthetic_df[FEATURE_NAMES], synthetic_df[TARGET].astype(int).to_numpy()


# ---------------------------------------------------------------- training
@pytest.mark.parametrize("model_type, hp", [
    ("logistic_regression", {"C": 1.0}),
    ("random_forest", {"n_estimators": 10, "max_depth": 4}),
    ("xgboost", {"n_estimators": 20, "max_depth": 3}),
])
def test_each_model_type_trains_and_predicts(xy, model_type, hp):
    X, y = xy
    pipe = build_pipeline(model_type, hp, pos_weight=5.0, seed=0).fit(X, y)
    proba = pipe.predict_proba(X.head(10))[:, 1]
    assert proba.shape == (10,) and ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_pipeline("svm", {}, 1.0, 0)


def test_higher_beta_gives_higher_recall(xy):
    X, y = xy
    proba = build_pipeline("logistic_regression", {"C": 1.0}, 1.0, 0).fit(X, y).predict_proba(X)[:, 1]
    r1 = classification_metrics(y, proba, tune_threshold(y, proba, 1.0))["recall"]
    r3 = classification_metrics(y, proba, tune_threshold(y, proba, 3.0))["recall"]
    assert r3 >= r1


def test_metrics_contain_required_keys(xy):
    X, y = xy
    m = classification_metrics(y, np.full(len(y), 0.5), 0.5)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_feature_importance_is_normalised(xy):
    X, y = xy
    pipe = build_pipeline("random_forest", {"n_estimators": 10, "max_depth": 4}, 1.0, 0).fit(X, y)
    imp = feature_importance(pipe)
    assert abs(imp.sum() - 1) < 1e-9 and len(imp) == 25


# ---------------------------------------------------------------- acceptance gate
def test_acceptance_passes_and_fails():
    crit = {"min_roc_auc": 0.78, "min_recall": 0.70}
    assert check_acceptance({"roc_auc": 0.80, "recall": 0.75}, crit)["passed"]
    res = check_acceptance({"roc_auc": 0.80, "recall": 0.69}, crit)
    assert not res["passed"] and not res["checks"]["min_recall"]["passed"]


# ---------------------------------------------------------------- drift
def test_psi_is_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_no_drift_on_same_distribution(synthetic_df):
    ref = build_reference(synthetic_df.head(1500))
    report = compute_drift(ref, synthetic_df.tail(1500), min_samples=100)
    assert report["status"] == "ok" and not report["drift_detected"]


def test_drift_detected_when_bmi_shifts(synthetic_df):
    ref = build_reference(synthetic_df)
    shifted = synthetic_df.copy()
    shifted["BMI"] = shifted["BMI"] + 12
    report = compute_drift(ref, shifted, min_samples=100)
    assert report["drift_detected"] and "BMI" in report["drifted_features"]


def test_drift_needs_minimum_samples(synthetic_df):
    report = compute_drift(build_reference(synthetic_df), synthetic_df.head(10), min_samples=100)
    assert report["status"] == "insufficient_data" and not report["drift_detected"]


# ---------------------------------------------------------------- explanations
def test_what_if_only_changes_modifiable_unhealthy_factors(patient):
    rows, changed = what_if_rows({k: float(v) for k, v in patient.items()})
    names = [f.name for f, _, _ in changed]
    assert "Age" not in names and "Sex" not in names       # not modifiable
    assert "Veggies" not in names                          # already healthy
    assert {"HighBP", "BMI", "PhysActivity"} <= set(names)
    assert len(rows) == len(changed) + 1


def test_low_bmi_is_never_pushed_up(patient):
    p = {k: float(v) for k, v in patient.items()}
    p["BMI"] = 19.0
    _, changed = what_if_rows(p)
    assert "BMI" not in [f.name for f, _, _ in changed]


def test_top_factors_sorted_and_filtered(patient):
    _, changed = what_if_rows({k: float(v) for k, v in patient.items()})
    scores = np.array([0.8] + [0.8 - 0.01 * (i + 1) for i in range(len(changed))])
    scores[1] = 0.799                                        # below MIN_IMPACT -> dropped
    out = top_factors(scores, changed, k=3)
    assert len(out) == 3
    assert [f["impact"] for f in out] == sorted([f["impact"] for f in out], reverse=True)
    assert changed[0][0].name not in [f["feature"] for f in out]
