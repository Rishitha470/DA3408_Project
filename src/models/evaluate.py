"""Stage 4 - evaluate: score the selected model on the held-out test split, apply the acceptance
gate and, if it passes, register the model in the MLflow Model Registry with alias 'champion'.

Outputs (tracked by DVC): reports/metrics.json, reports/acceptance.json, reports/plots/*.csv
Run:  python -m src.models.evaluate
"""
import json
import sys

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.common.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, load_params, tracking_uri
from src.common.logging_utils import get_logger
from src.common.pipeline_log import track_stage
from src.common.reproducibility import git_commit
from src.common.schema import FEATURE_NAMES, TARGET
from src.models.train import classification_metrics, plot_confusion, plot_roc

log = get_logger("evaluate")


class AcceptanceError(Exception):
    """Raised when the model does not meet the acceptance criteria."""


def check_acceptance(metrics: dict, criteria: dict) -> dict:
    """Compare test metrics with the minimums in params.yaml -> evaluate.acceptance."""
    checks = {name: {"required": req, "actual": round(metrics[name.replace("min_", "")], 4),
                     "passed": metrics[name.replace("min_", "")] >= req}
              for name, req in criteria.items()}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def write_plot_data(y, proba, threshold) -> None:
    """CSV files for `dvc plots show` (ROC, precision-recall, confusion matrix)."""
    plots = REPORTS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, proba)
    step = max(1, len(fpr) // 300)                       # keep files small
    pd.DataFrame({"fpr": fpr[::step], "tpr": tpr[::step]}).to_csv(plots / "roc.csv", index=False)
    prec, rec, _ = precision_recall_curve(y, proba)
    step = max(1, len(prec) // 300)
    pd.DataFrame({"recall": rec[::step], "precision": prec[::step]}).to_csv(plots / "pr.csv", index=False)
    pd.DataFrame({"actual": y, "predicted": (proba >= threshold).astype(int)}).to_csv(
        plots / "confusion.csv", index=False)


def register(meta: dict, metrics: dict, model_name: str) -> str:
    """Register the run's model and point alias 'champion' to it. Returns the version."""
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{meta['run_id']}/model", model_name)
    client.set_registered_model_alias(model_name, "champion", mv.version)
    for k, v in {"git_commit": git_commit(), "model": meta["model_name"],
                 "test_roc_auc": f"{metrics['roc_auc']:.4f}",
                 "test_recall": f"{metrics['recall']:.4f}"}.items():
        client.set_model_version_tag(model_name, mv.version, k, v)
    return mv.version


def main() -> int:
    params = load_params()
    cfg = params["evaluate"]
    mlflow.set_tracking_uri(tracking_uri())
    with track_stage("evaluate") as rec:
        bundle = joblib.load(MODELS_DIR / "model.joblib")
        meta, threshold = bundle["meta"], bundle["threshold"]
        test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
        X, y = test[FEATURE_NAMES].astype(float), test[TARGET].astype(int).to_numpy()
        rec.set_rows(len(test))

        proba = bundle["pipeline"].predict_proba(X)[:, 1]
        metrics = {k: round(float(v), 4) for k, v in classification_metrics(y, proba, threshold).items()}
        tn, fp, fn, tp = confusion_matrix(y, (proba >= threshold).astype(int)).ravel()
        metrics.update(threshold=round(threshold, 4), true_negatives=int(tn), false_positives=int(fp),
                       false_negatives=int(fn), true_positives=int(tp), n_test=int(len(y)))
        acceptance = check_acceptance(metrics, cfg["acceptance"])
        write_plot_data(y, proba, threshold)

        # attach the test results to the selected model's MLflow run
        with mlflow.start_run(run_id=meta["run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_figure(plot_confusion(y, proba, threshold, "Test confusion matrix"),
                              "plots/confusion_matrix_test.png")
            mlflow.log_figure(plot_roc(y, proba, "Test ROC"), "plots/roc_test.png")
            mlflow.log_dict(acceptance, "acceptance.json")
            mlflow.set_tag("acceptance", "passed" if acceptance["passed"] else "failed")

        out = {"model": meta["model_name"], "model_version": meta["model_version"], **metrics}
        if acceptance["passed"]:
            version = register(meta, metrics, cfg["registered_model_name"])
            out["registry"] = {"name": cfg["registered_model_name"], "version": version,
                               "alias": "champion"}
            log.info("ACCEPTED: registered %s version %s as @champion",
                     cfg["registered_model_name"], version)
        (REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
        (REPORTS_DIR / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
        log.info("test metrics: %s", {k: metrics[k] for k in ("roc_auc", "recall", "precision", "f1")})
        if not acceptance["passed"]:
            failed = [k for k, c in acceptance["checks"].items() if not c["passed"]]
            raise AcceptanceError(f"model rejected, failed criteria: {failed}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AcceptanceError as exc:
        log.error("%s", exc)
        sys.exit(1)
