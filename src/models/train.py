"""Stage 3 - train: train candidate models, track every run in MLflow, select the best one.

For each candidate in params.yaml (Logistic Regression, Random Forest, XGBoost and their
hyper-parameter variants) a nested MLflow run logs parameters, validation metrics, training time,
inference latency, a confusion matrix and an ROC curve. Class imbalance is handled with class
weights (LR/RF) and scale_pos_weight (XGBoost); the decision threshold is tuned on the validation
split to maximise F-beta (beta > 1 favours recall, i.e. fewer missed at-risk people).
Run:  python -m src.models.train
"""
import json
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, Tuple

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src.common.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, ROOT, load_params, tracking_uri
from src.common.features import add_features
from src.common.logging_utils import get_logger
from src.common.pipeline_log import track_stage
from src.common.reproducibility import file_md5, git_commit, git_dirty
from src.common.schema import FEATURE_NAMES, TARGET

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("train")


# ---------------------------------------------------------------- model construction
def build_pipeline(model_type: str, hp: Dict, pos_weight: float, seed: int) -> Pipeline:
    """Feature engineering + (scaling) + classifier, all in one sklearn Pipeline."""
    steps = [("features", FunctionTransformer(add_features, validate=False))]
    if model_type == "logistic_regression":
        steps.append(("scale", StandardScaler()))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, **hp)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=seed, **hp)
    elif model_type == "xgboost":
        clf = XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1,
                            tree_method="hist", random_state=seed, **hp)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    steps.append(("clf", clf))
    return Pipeline(steps)


# ---------------------------------------------------------------- metrics
def tune_threshold(y_true: np.ndarray, proba: np.ndarray, beta: float) -> float:
    """Threshold maximising F-beta on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p, r = precision[:-1], recall[:-1]
    fbeta = (1 + beta ** 2) * p * r / np.clip(beta ** 2 * p + r, 1e-12, None)
    return float(thresholds[int(np.argmax(fbeta))])


def classification_metrics(y_true, proba, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def plot_confusion(y_true, proba, threshold, title):
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_roc(y_true, proba, title):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def feature_importance(pipeline: Pipeline) -> pd.Series:
    """Global importance: |coefficient| for LR, built-in importance for tree models."""
    names = add_features(pd.DataFrame([dict.fromkeys(FEATURE_NAMES, 1.0)])).columns
    clf = pipeline.named_steps["clf"]
    values = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    s = pd.Series(values, index=names)
    return (s / s.sum()).sort_values(ascending=False)


def single_row_latency_ms(pipeline: Pipeline, X: pd.DataFrame, n: int = 100) -> float:
    """p95 latency of single-row predictions (what one web request costs)."""
    times = []
    for i in range(min(n, len(X))):
        t0 = time.perf_counter()
        pipeline.predict_proba(X.iloc[[i]])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


# ---------------------------------------------------------------- one candidate
def train_candidate(model_type, hp, data, beta, seed) -> Tuple[Pipeline, Dict, np.ndarray, float]:
    X_tr, y_tr, X_val, y_val = data
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    pipe = build_pipeline(model_type, hp, pos_weight, seed)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    proba = pipe.predict_proba(X_val)[:, 1]
    thr = tune_threshold(y_val, proba, beta)
    metrics = classification_metrics(y_val, proba, thr)
    metrics.update(threshold=thr, train_time_s=train_s,
                   latency_p95_ms=single_row_latency_ms(pipe, X_val))
    return pipe, metrics, proba, thr


def load_split(name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(DATA_PROCESSED / f"{name}.parquet")
    return df[FEATURE_NAMES].astype(float), df[TARGET].astype(int).to_numpy()


def main() -> None:
    params = load_params()
    cfg, seed = params["train"], params["seed"]
    np.random.seed(seed)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(cfg["experiment_name"])

    with track_stage("train") as rec:
        X_tr, y_tr = load_split("train")
        X_val, y_val = load_split("val")
        data = (X_tr, y_tr, X_val, y_val)
        rec.set_rows(len(X_tr))
        data_hash = file_md5(DATA_PROCESSED / "train.parquet")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        results, best = [], None
        with mlflow.start_run(run_name=f"training-{stamp}") as parent:
            mlflow.set_tags({"git_commit": git_commit(), "git_dirty": str(git_dirty()),
                             "train_data_md5": data_hash, "stage": "train"})
            mlflow.log_params({"seed": seed, "selection_metric": cfg["selection_metric"],
                               "threshold_beta": cfg["threshold_beta"], "n_train": len(X_tr),
                               "n_val": len(X_val), "positive_rate_train": round(float(y_tr.mean()), 4)})
            mlflow.log_artifact(str(ROOT / "params.yaml"))
            if (REPORTS_DIR / "data_quality.json").exists():
                mlflow.log_artifact(str(REPORTS_DIR / "data_quality.json"))

            for model_type, configs in cfg["candidates"].items():
                for i, hp in enumerate(configs):
                    name = f"{model_type}_{i}"
                    with mlflow.start_run(run_name=name, nested=True) as child:
                        mlflow.set_tags({"model_type": model_type, "git_commit": git_commit(),
                                         "train_data_md5": data_hash})
                        mlflow.log_params({"model_type": model_type, **hp})
                        pipe, metrics, proba, thr = train_candidate(model_type, hp, data,
                                                                    cfg["threshold_beta"], seed)
                        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                        mlflow.log_figure(plot_confusion(y_val, proba, thr, f"{name} (val)"),
                                          "plots/confusion_matrix_val.png")
                        mlflow.log_figure(plot_roc(y_val, proba, f"{name} (val)"), "plots/roc_val.png")
                        plt.close("all")
                        log.info("%-22s val roc_auc=%.4f recall=%.3f precision=%.3f thr=%.3f (%.1fs)",
                                 name, metrics["roc_auc"], metrics["recall"], metrics["precision"],
                                 thr, metrics["train_time_s"])
                        row = {"name": name, "model_type": model_type, "params": hp,
                               "run_id": child.info.run_id,
                               **{k: round(float(v), 4) for k, v in metrics.items()}}
                        results.append(row)
                        if best is None or metrics[cfg["selection_metric"]] > best[1][cfg["selection_metric"]]:
                            best = (row, metrics, pipe)

            best_row, best_metrics, best_pipe = best
            importance = feature_importance(best_pipe)
            save_artifacts(best_row, best_metrics, best_pipe, importance, X_val, params, data_hash,
                           parent.info.run_id)
            mlflow.set_tags({"best_model": best_row["name"], "best_run_id": best_row["run_id"]})
            mlflow.log_metric(f"best_val_{cfg['selection_metric']}", best_metrics[cfg["selection_metric"]])
            mlflow.log_dict({"candidates": results}, "model_comparison.json")

        comparison = {"selection_metric": cfg["selection_metric"], "best": best_row["name"],
                      "candidates": results}
        (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
        rec["best_model"] = best_row["name"]
        log.info("best model: %s (val %s = %.4f)", best_row["name"], cfg["selection_metric"],
                 best_metrics[cfg["selection_metric"]])


def save_artifacts(best_row, metrics, pipe, importance, X_val, params, data_hash, parent_run_id):
    """Write models/ (joblib bundle, metadata, MLflow pyfunc model) and log the model to MLflow."""
    from mlflow.models import infer_signature
    from src.models.pyfunc import DiabetesRiskPyfunc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = f"{best_row['name']}@{best_row['run_id'][:8]}"
    meta = {
        "model_name": best_row["name"], "model_type": best_row["model_type"],
        "model_version": version, "run_id": best_row["run_id"], "parent_run_id": parent_run_id,
        "params": best_row["params"], "threshold": float(metrics["threshold"]),
        "val_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "features": FEATURE_NAMES, "feature_importance": importance.round(4).to_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "train_data_md5": data_hash,
    }
    reference = json.loads((DATA_PROCESSED / "reference.json").read_text())
    bundle = {"pipeline": pipe, "threshold": meta["threshold"], "model_version": version,
              "meta": meta, "reference": reference}
    joblib.dump(bundle, MODELS_DIR / "model.joblib", compress=3)
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (MODELS_DIR / "drift_reference.json").write_text(json.dumps(reference))

    # deployable MLflow model (served by `mlflow models serve` in Docker)
    example = X_val.head(3)
    pyfunc = DiabetesRiskPyfunc()

    class _Ctx:
        artifacts = {"bundle": str(MODELS_DIR / "model.joblib")}

    pyfunc.load_context(_Ctx())
    signature = infer_signature(example, pyfunc.predict(None, example))
    kwargs = dict(python_model=DiabetesRiskPyfunc(), artifacts={"bundle": str(MODELS_DIR / "model.joblib")},
                  code_paths=[str(ROOT / "src")], signature=signature, input_example=example,
                  pip_requirements=str(ROOT / "docker" / "requirements-model.txt"))
    out_dir = MODELS_DIR / "mlflow_model"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    mlflow.pyfunc.save_model(path=str(out_dir), **kwargs)

    # same model logged into the winning child run, so it can be registered from the run
    with mlflow.start_run(run_id=best_row["run_id"], nested=True):
        mlflow.pyfunc.log_model(name="model", **kwargs)
        fig, ax = plt.subplots(figsize=(5, 5))
        importance.head(15)[::-1].plot.barh(ax=ax)
        ax.set_title("Feature importance (normalised)")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/feature_importance.png")
        mlflow.log_dict(meta["feature_importance"], "feature_importance.json")
        mlflow.set_tag("selected", "true")
        plt.close("all")


if __name__ == "__main__":
    main()
