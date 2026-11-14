# High-Level Design

## 1. Problem and success metrics

Many people with diabetes or prediabetes do not know it. The application gives a quick, free
**screening** estimate from 21 self-reported answers (CDC BRFSS 2015 survey indicators) and points
people at risk towards a blood test. It is explicitly not a diagnosis.

| Goal | Metric | Target |
|---|---|---|
| Do not miss people at risk | Recall on the held-out test set | >= 0.70 |
| Separate risk well overall | ROC-AUC on the test set | >= 0.78 |
| Avoid flagging almost everyone | Precision on the test set | >= 0.25 |
| Responsive app | p95 latency of `/api/v1/predict` | < 500 ms (alert) |
| Reliability | 5xx share of predictions | < 5 % (alert) |

Recall is prioritised because a missed at-risk person is worse than an unnecessary blood test.

## 2. Data

* **Source:** CDC Diabetes Health Indicators, UCI ML Repository id 891, 253,680 survey responses,
  21 features, binary target (diabetes/prediabetes vs. none). Public and anonymous.
* **Imbalance:** about 14 % positive. Handled with class weights / `scale_pos_weight` and a tuned threshold,
  not with resampling (keeps the data distribution intact for drift monitoring).
* **Known biases:** self-reported answers; US population only; income in US dollars; prediabetes is merged
  with diabetes. These are stated in the user manual.
* **Duplicates:** identical rows are kept (their count is reported in `reports/data_quality.json`), because identical survey answers from different people are genuine
  (configurable: `process.drop_duplicates`).

## 3. Key design choices and rationale

| Decision | Choice | Why | Alternative considered |
|---|---|---|---|
| Data engineering | Ray Data | Parallel block processing on a laptop, no cluster needed; same code scales out. | Spark: heavier JVM setup for 250k rows. |
| Pipeline / CI | DVC DAG + `scripts/ci.sh` | Re-runs only changed stages; data and models versioned next to the code; metrics diff between commits. | Cloud CI services are not allowed. |
| Experiment tracking | MLflow with nested runs | One parent run per training, one child per candidate; Git commit and data hash tags make every result reproducible. | |
| Model choice | Best of LR / RF / XGBoost by validation ROC-AUC | Simple, explainable baselines vs. strong tabular models. Selection is automatic and logged. | |
| Threshold | Maximise F-beta, beta = 2 | Encodes "recall matters twice as much as precision". | Default 0.5 would miss many positives after class weighting. |
| Feature engineering | Inside the sklearn Pipeline | The exact same code runs in training and serving: no training/serving skew. | Separate preprocessing step in the API. |
| Serving | MLflow model server behind a FastAPI backend | MLflow gives a standard, versioned inference API; FastAPI adds validation, explanation, metrics and UI endpoints. The two are separate containers. | Model inside the API process (still available as `INFERENCE_MODE=local`). |
| Loose coupling | UI -> REST -> backend -> REST -> model server | Each block can be replaced or scaled independently. The UI only knows `API_BASE_URL`. | |
| Explanation | Model-agnostic "what-if" on changeable factors | Works for any model, gives actionable advice, costs one batched call. | SHAP: heavier, less actionable for lay users. |
| Drift | PSI per feature over the 500 most recent predictions, live in the backend + batch job | Simple, well-understood thresholds (0.1 / 0.2); visible in Grafana in near real time. A rolling window of recent requests reacts to a shift within a few hundred requests; a larger window (5,000 was tried) diluted a real shift below the alarm level. | KS test: less interpretable for categorical codes. |
| Security | Encrypted input log (Fernet), no personal identifiers, backend container runs as a non-root user, CORS allow-list, admin token for reload | Health data is sensitive. | |
| Packaging | Docker Compose: frontend, backend, model-server, MLflow, Prometheus, Alertmanager, Grafana, node-exporter | One command starts everything; identical environments. | Kubernetes: unnecessary for one machine. |

## 4. Software paradigm

The code is **object-oriented where there is state or substitutable behaviour** and functional elsewhere:

* `BasePredictor` (abstract) with `LocalPredictor` and `MlflowServerPredictor` (Strategy pattern, chosen by
  `make_predictor` factory from `INFERENCE_MODE`).
* `PredictionStore` encapsulates the encrypted log, the recent-input window and drift computation.
* `Settings` (dataclass) holds all configuration from environment variables.
* Pipeline stages, metrics and drift maths are pure functions, which makes them easy to unit test.

## 5. Lifecycle (AIOps)

1. **Develop:** change code or `params.yaml` -> `scripts/ci.sh` (lint, 74 tests, `dvc repro`, metrics diff).
2. **Train:** DVC runs only the affected stages; every run is in MLflow with the Git commit.
3. **Gate:** the evaluate stage registers the model as `@champion` only if acceptance criteria hold.
4. **Deploy:** `docker compose up -d --build`; `models/` is mounted into the model server and the backend.
   After retraining, both containers are restarted (DVC recreates `models/`, and a Docker bind mount keeps
   pointing at the deleted folder until the container restarts).
5. **Monitor:** Prometheus + Grafana + alert rules; live PSI drift per feature.
6. **Retrain:** `python -m src.monitoring.drift_monitor --retrain` runs `dvc repro --force` when drift is found.
7. **Roll back:** `scripts/rollback.sh <commit>` restores the earlier `models/` from the DVC cache and the
   matching reports, moves the registry alias `@champion` back to that version, and restarts the services.

## 6. Reproducibility

Every MLflow run carries `git_commit`, `git_dirty` and `train_data_md5` tags, and every registered model
version carries the `git_commit` it was built from. `dvc.lock` records the hash of every input and output.
To reproduce a result: `git checkout <commit> && dvc checkout` (or `dvc repro`). Seeds are fixed in `params.yaml`.

Ray Data processes blocks in parallel and does not guarantee row order, so the cleaned rows are sorted into a
canonical order before the seeded train/validation/test split (`split_data()`, test TC-DATA-10). Without this,
the same seed selected different people on each run (found during testing, see section 8.7).

## 7. Limitations

* The score ranks risk; it is not a calibrated probability (class weighting shifts scores upwards). The UI says so.
* The model learned from US survey data from 2015 and may transfer imperfectly to other populations.
* Retraining on drift re-ingests the public dataset; in a real deployment newly labelled data would be added.
  (Retraining on the same data therefore gives the same scores, see 8.5.)

## 8. Results

All numbers below come from one machine: a VirtualBox VM with Ubuntu 26.04, 6 vCPUs, 7.2 GB RAM,
Python 3.12.16 and Docker 29.1 (host laptop: Intel Core i5-1235U), on 2026-11-14. They can be regenerated with
`dvc repro`, `dvc metrics show` and the scripts named in each section.

### 8.1 Data pipeline (`reports/ingest.json`, `reports/data_quality.json`)

| Item | Value |
|---|---|
| Source | UCI id 891, 253,680 rows, 22 columns, MD5 `2500563d61be116380f818e68b6a7083` |
| Rows in / kept / invalid | 253,680 / 253,680 / 0 |
| Positive rate (diabetes or prediabetes) | 13.93 % |
| Duplicate rows (kept on purpose) | 24,206 |
| Split train / validation / test | 177,576 / 38,052 / 38,052 (stratified 70/15/15) |
| Ray (4 CPUs, 8 blocks): read / validate / clean | 7.2 s / 2.7 s / 1.3 s |
| Throughput | 22,723 rows/s for the Ray processing; 14,194 rows/s for the whole stage incl. Ray start-up (17.9 s) |

### 8.2 Model comparison on the validation split (`reports/model_comparison.json`)

| Candidate | Hyperparameters | ROC-AUC | Recall | Precision | F1 | Threshold | Train s | p95 ms |
|---|---|---|---|---|---|---|---|---|
| logistic_regression_0 | C=0.1 | 0.8256 | 0.828 | 0.289 | 0.428 | 0.433 | 2.5 | 7.0 |
| logistic_regression_1 | C=1.0 | 0.8256 | 0.828 | 0.289 | 0.428 | 0.433 | 3.0 | 6.9 |
| random_forest_0 | 100 trees, depth 10, leaf 50 | 0.8267 | 0.846 | 0.283 | 0.424 | 0.433 | 5.0 | 124.8 |
| random_forest_1 | 150 trees, depth 12, leaf 100 | 0.8268 | 0.833 | 0.287 | 0.427 | 0.448 | 7.7 | 167.5 |
| **xgboost_0 (selected)** | 300 trees, depth 4, lr 0.1 | **0.8299** | 0.807 | 0.304 | 0.442 | 0.480 | 1.0 | 7.8 |
| xgboost_1 | 500 trees, depth 6, lr 0.05 | 0.8271 | 0.823 | 0.292 | 0.431 | 0.447 | 2.3 | 22.8 |

All six candidates lie within 0.005 ROC-AUC of each other: performance is limited by the 21 self-reported
answers more than by the algorithm. XGBoost was selected (highest validation ROC-AUC) and is also among the
fastest to train and to score.

### 8.3 Held-out test set (`reports/metrics.json`, `reports/acceptance.json`)

| Metric | Value | Acceptance |
|---|---|---|
| ROC-AUC | 0.834 | >= 0.78 PASS |
| Recall | 0.818 | >= 0.70 PASS |
| Precision | 0.301 | >= 0.25 PASS |
| F1 / PR-AUC / accuracy | 0.440 / 0.440 / 0.710 | - |
| Confusion matrix (38,052 people) | 4,335 true positives, 967 missed, 10,063 false alarms, 22,687 true negatives | - |

Of 5,302 people with diabetes or prediabetes in the test set, the model flags 4,335 (82 %). The price is about
two false alarms per correct alarm, which is acceptable for a screening tool whose follow-up is a cheap blood
test. Accuracy (0.71) is below the 0.86 of a model that always answers "no diabetes"; that is why accuracy is
not a target metric.

### 8.4 Serving performance (`scripts/load_test.py`, 10 s per level)

| Concurrent users | Requests | Errors | Requests/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| 1 | 301 | 0 | 30.1 | 32.0 | 43.6 | 54.5 |
| 4 | 267 | 0 | 26.7 | 141.2 | 231.5 | 474.7 |
| 8 | 239 | 0 | 23.9 | 332.3 | 516.5 | 633.7 |

Each request includes one batched model call for the person and the what-if rows of the explanation. The
target (p95 < 500 ms at 4 users) is met. At 8 users p95 is just above 500 ms: the load generator, the backend,
the model server and six other containers share the same 6 vCPUs, so throughput does not grow with more users.

### 8.5 Monitoring, drift, retraining and rollback

| Scenario | Observed result |
|---|---|
| 500 normal requests (real test rows) | drift not detected, max PSI 0.020 |
| 400 shifted requests (older, heavier, less active) | drift detected on BMI, PhysActivity and Age, max PSI 1.77; alert `DataDriftDetected` firing within 90 s |
| Model server stopped | `POST /api/v1/predict` returns HTTP 503 `model_unavailable` with a friendly message; alert `ModelServerDown` firing; `docker compose start model-server` recovers |
| `drift_monitor --retrain` | drift confirmed from the encrypted log, `dvc repro --force` (about 3 min), model accepted and registered as version 4 `@champion` (test ROC-AUC 0.8341, identical: same data) |
| `scripts/rollback.sh HEAD~1` | previous model restored from the DVC cache in seconds without retraining; `@champion` moved back to version 3; serving `xgboost_0@01297897` |

### 8.6 Reproducibility evidence

Two complete `dvc repro --force` runs from the same commit produced identical validation metrics for all six
candidates and identical test metrics (registry versions 2 and 3, both ROC-AUC 0.8341). A third, drift-triggered
retraining (version 4) again gave identical metrics. Version 1, produced before the canonical-order fix, scored
0.8300: the difference that revealed the problem.

### 8.7 Problems found during testing and how they were fixed

| Problem | Cause | Fix |
|---|---|---|
| Different scores from the same seed | Ray returned rows in a different order on each run | canonical sort before the split; TC-DATA-10 |
| Drift not detected for a clear shift | 5,000-request window diluted 400 shifted requests | window of the 500 most recent predictions (`monitoring.drift_window`) |
| Retraining failed at model registration | under heavy CPU load the single MLflow worker was restarted mid-request | 2 MLflow workers; idempotent registration with retries; TC-MOD-14 |
| Backend could not find the model after retraining | DVC recreates `models/`; the Docker bind mount kept the deleted folder | restart model server and backend after retraining (`rollback.sh` does this) |
| Healthy values shown in red, "No data" for errors | Grafana default thresholds; empty error series | neutral colours for informational panels; `or vector(0)` in the error rule and panel |
| Containers could not read files on the VM shared folder | VirtualBox shared folder mounted as root:vboxsf, mode 770 | mount with `uid=1000,gid=1000,dmode=777,fmode=777` (deployment note in the README) |
