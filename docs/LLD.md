# Low-Level Design

## 1. REST API (backend, FastAPI, port 8000)

Base URL is configurable; the frontend reads it from `API_BASE_URL`. Interactive specification:
`http://localhost:8000/docs` (OpenAPI generated from the code). All responses carry an `X-Request-ID`
header (echoed if the client sends one).

| Method | Path | Purpose | Success | Errors |
|---|---|---|---|---|
| GET | `/health` | Liveness, model status, dependency health | 200 | - |
| GET | `/ready` | Readiness (model loaded) | 200 | 503 |
| POST | `/api/v1/predict` | Risk prediction for one person | 200 | 422, 503, 500 |
| GET | `/api/v1/model` | Model card (version, threshold, metrics, importance) | 200 | 503 |
| POST | `/api/v1/model/reload` | Reload model after retraining (header `X-Admin-Token`) | 200 | 401, 503 |
| GET | `/api/v1/pipeline` | Pipeline console data | 200 | - |
| GET | `/api/v1/drift` | Current drift report | 200 | - |
| GET | `/metrics` | Prometheus metrics (text format) | 200 | - |

### 1.1 `POST /api/v1/predict`

**Request** (`application/json`, all fields required, unknown fields rejected):

| Field | Type | Range | Meaning |
|---|---|---|---|
| HighBP | int | 0-1 | Told by a doctor: high blood pressure |
| HighChol | int | 0-1 | Told by a doctor: high cholesterol |
| CholCheck | int | 0-1 | Cholesterol check in last 5 years |
| BMI | float | 12-98 | Body mass index |
| Smoker | int | 0-1 | Smoked 100+ cigarettes in life |
| Stroke | int | 0-1 | Ever had a stroke |
| HeartDiseaseorAttack | int | 0-1 | Coronary heart disease or heart attack |
| PhysActivity | int | 0-1 | Physical activity in past 30 days |
| Fruits | int | 0-1 | Fruit 1+ times per day |
| Veggies | int | 0-1 | Vegetables 1+ times per day |
| HvyAlcoholConsump | int | 0-1 | Heavy drinking |
| AnyHealthcare | int | 0-1 | Has health coverage |
| NoDocbcCost | int | 0-1 | Skipped doctor because of cost |
| GenHlth | int | 1-5 | General health (1 excellent ... 5 poor) |
| MentHlth | int | 0-30 | Days of poor mental health (last 30) |
| PhysHlth | int | 0-30 | Days of poor physical health (last 30) |
| DiffWalk | int | 0-1 | Difficulty walking/climbing stairs |
| Sex | int | 0-1 | 0 female, 1 male |
| Age | int | 1-13 | Age group (1 = 18-24 ... 13 = 80+) |
| Education | int | 1-6 | Education level |
| Income | int | 1-8 | Income level |

**Response 200** (example; values depend on the trained model):

```json
{
  "request_id": "7076f6955121466193b791fb98201636",
  "probability": 0.937,
  "prediction": 1,
  "risk_level": "high",
  "threshold": 0.5202,
  "top_factors": [
    {"feature": "GenHlth", "label": "General health (1 = excellent, 5 = poor)",
     "current_value": 4.0, "healthier_value": 2.0, "impact": 0.1426}
  ],
  "model_name": "xgboost_0",
  "model_version": "xgboost_0@8b32ed00",
  "disclaimer": "This is a screening and awareness tool, not a medical diagnosis. ..."
}
```

Rules: `prediction = 1` if `probability >= threshold`; `risk_level` is `high` at or above the threshold,
`moderate` from `0.6 x threshold`, otherwise `low`. `top_factors` has at most 3 entries, sorted by `impact`
(drop in score if the factor were at its healthier value); only changeable factors are considered.

**Error body** (all errors):

```json
{"error": "validation_failed", "message": "Some answers are missing or out of range.",
 "request_id": "...", "details": [{"field": "BMI", "msg": "Input should be greater than or equal to 12"}]}
```

| Status | `error` | When |
|---|---|---|
| 422 | `validation_failed` | Missing/unknown field, wrong type, out of range, malformed JSON |
| 503 | `model_unavailable` | Model not loaded or model server unreachable / bad response |
| 500 | `internal_error` | Any unexpected error (no internal details are returned) |
| 401 | `unauthorized` | Wrong admin token on reload |

### 1.2 `GET /api/v1/pipeline` (response keys)

`dag` (`stages[]` with `name`, `cmd`, `deps`, `last_run`; `edges[]` as `[from, to]`), `other_jobs`,
`runs[]` (last 50 stage executions: `stage`, `status`, `started_at`, `finished_at`, `duration_s`, `rows_in`,
`rows_out`, `rows_per_s`, `error`), `summary` (`total`, `success`, `failed`), `data_quality`,
`model_comparison`, `test_metrics`, `acceptance`, `drift`, `tools` (name -> URL).

### 1.3 `GET /api/v1/drift`

`{"computed_at", "n_samples", "min_samples", "status": "ok|insufficient_data|no_reference",
"drift_detected": bool, "drifted_features": [...], "max_psi", "features": {"BMI": {"psi": 0.62, "level": "significant"}, ...}}`

## 2. Model server (MLflow, port 5001)

`POST /invocations`, body `{"dataframe_split": {"columns": [21 feature names], "data": [[...], ...]}}`.
Response `{"predictions": [{"probability", "prediction", "risk_level", "model_version"}, ...]}`.
`GET /ping` -> 200 when ready. Signature and input example are stored in `models/mlflow_model/MLmodel`.

## 3. Modules and classes

| Module | Contents |
|---|---|
| `src/common/schema.py` | `Feature` dataclass, `FEATURES` (ranges, labels, healthy values, modifiable flag), `risk_level()` |
| `src/common/features.py` | `add_features()` - BMI category, cardio risk count, lifestyle score, poor-health days |
| `src/common/drift.py` | `build_reference()`, `psi()`, `compute_drift()` |
| `src/common/security.py` | `get_cipher()`, `append_encrypted()`, `read_encrypted()` (Fernet) |
| `src/common/pipeline_log.py` | `track_stage()` context manager, `read_runs()` |
| `src/common/reproducibility.py` | `git_commit()`, `git_dirty()`, `file_md5()` |
| `src/data/ingest.py` | `load_source()`, `main()` |
| `src/data/process.py` | `invalid_mask()`, `batch_quality()`, `clean_batch()`, `run_ray_pipeline()`, `quality_gate()` |
| `src/models/train.py` | `build_pipeline()`, `tune_threshold()`, `classification_metrics()`, `train_candidate()`, `save_artifacts()` |
| `src/models/pyfunc.py` | `DiabetesRiskPyfunc` (MLflow PythonModel) |
| `src/models/evaluate.py` | `check_acceptance()`, `write_plot_data()`, `register()` |
| `src/monitoring/drift_monitor.py` | batch drift check and retraining trigger |
| `backend/app/config.py` | `Settings` (all env vars) |
| `backend/app/predictor.py` | `BasePredictor` -> `LocalPredictor`, `MlflowServerPredictor`; `make_predictor()`; `PredictorError` |
| `backend/app/explain.py` | `what_if_rows()`, `top_factors()` |
| `backend/app/prediction_store.py` | `PredictionStore` |
| `backend/app/metrics.py` | Prometheus metric objects and gauge updaters |
| `backend/app/pipeline_status.py` | `load_dag()`, `build_status()` |
| `backend/app/main.py` | FastAPI app, middleware, exception handlers, endpoints, background refresher |

## 4. Configuration (environment variables of the backend)

| Variable | Default | Meaning |
|---|---|---|
| `INFERENCE_MODE` | `local` (Compose: `mlflow`) | `mlflow` = call model server, `local` = load joblib |
| `MODEL_SERVER_URL` | `http://localhost:5001/invocations` | Model server endpoint |
| `MODEL_DIR`, `REPORTS_DIR`, `LOG_DIR`, `DVC_FILE` | repo paths | Mounted directories |
| `PREDICTION_LOG_KEY` | - | Fernet key for the encrypted input log |
| `CORS_ORIGINS` | `*` (Compose: UI origins) | Allowed browser origins |
| `REQUEST_TIMEOUT_S` | 5 | Timeout for model server calls |
| `REFRESH_INTERVAL_S` | 15 | Monitoring refresh period |
| `DRIFT_WINDOW`, `DRIFT_MIN_SAMPLES` | 5000, 100 | Drift window and minimum sample size |
| `ADMIN_TOKEN` | empty | Protects model reload |
| `DEPENDENCY_URLS` | empty | `name=url` health checks for `dependency_up` |
| `TOOL_LINKS` | localhost URLs | Links shown in the pipeline console |

## 5. Prometheus metrics

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | counter | method, endpoint, status |
| `http_request_duration_seconds` | histogram | method, endpoint |
| `http_requests_in_progress` | gauge | - |
| `app_exceptions_total` | counter | type |
| `model_predictions_total` | counter | risk_level |
| `model_prediction_probability` | histogram | - |
| `model_inference_duration_seconds` | histogram | - |
| `model_info`, `model_test_metric`, `model_decision_threshold` | gauge | model_name/version/mode, metric |
| `data_drift_psi`, `data_drift_detected`, `data_drift_window_samples` | gauge | feature |
| `pipeline_stage_last_status`, `..._last_duration_seconds`, `..._rows_per_second`, `..._last_run_timestamp_seconds`, `pipeline_stage_runs` | gauge | stage, status |
| `data_pipeline_rows`, `data_pipeline_ray_rows_per_second` | gauge | kind |
| `dependency_up` | gauge | service |

## 6. Files written by the pipeline

| Path | Written by | Tracked by |
|---|---|---|
| `data/raw/diabetes.csv` | ingest | DVC |
| `data/processed/{train,val,test}.parquet`, `reference.json` | process | DVC |
| `models/model.joblib`, `model_meta.json`, `drift_reference.json`, `mlflow_model/` | train | DVC |
| `reports/ingest.json`, `data_quality.json`, `model_comparison.json`, `metrics.json`, `acceptance.json`, `plots/*.csv` | stages | Git (DVC metrics/plots) |
| `reports/pipeline_runs.jsonl` | every stage | local only |
| `logs/predictions.enc` | backend | local only (encrypted) |
