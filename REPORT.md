# Diabetes Risk Prediction System: Project Report

In this report I go through what I built, how I handled each stage from data to monitoring, the numbers I got
when I ran it, and the problems I ran into along the way. The last section maps each point of the evaluation
guideline to where I covered it. I've kept the detailed design in the documents under [docs/](docs/) and just
link to them here, so this doesn't turn into a copy of the HLD.

---

## 1. Problem and objective

Diabetes often goes unnoticed until it causes complications, and not everyone can easily get screened. I built a
web application where a person answers 21 simple questions (age, BMI, blood pressure, physical activity,
general health and so on) and gets an estimate of their diabetes risk, together with the changes that would
lower it most. It is a **screening and awareness tool, not a diagnosis**, and the app says so on every result.

The objective was not only a working model but a complete AI system that is built, deployed, monitored and
maintained with AIOps practices: automated pipelines, version control of code, data and models, experiment
tracking, CI, containerised deployment, monitoring with alerts, drift detection, retraining and rollback.

### Success metrics (defined before training)

| Goal | Metric | Target | Achieved |
|---|---|---|---|
| Do not miss people at risk | Recall on the held-out test set | >= 0.70 | **0.818** |
| Rank risk well | ROC-AUC on the test set | >= 0.78 | **0.834** |
| Do not flag almost everyone | Precision on the test set | >= 0.25 | **0.301** |
| Responsive app | p95 latency of a prediction at 4 concurrent users | < 500 ms | **232 ms** |
| Reliability | Model outage handled without a crash | always | HTTP 503 with a friendly message |

Recall is the main metric because a missed at-risk person costs far more than an unnecessary blood test.

## 2. Data

* **Source:** CDC Diabetes Health Indicators (BRFSS 2015 survey), UCI ML Repository id 891: 253,680 responses,
  21 features, binary target (diabetes or prediabetes vs. none). Public and anonymous.
* **What the data-quality stage found:** no missing values, no out-of-range or invalid codes (0 rows dropped),
  13.93 % positives (strong class imbalance), and 24,206 duplicate rows. I kept the duplicates because identical
  answers from different people are genuine in a survey with mostly yes/no questions (configurable).
* **Known biases:** self-reported answers, US population only, income in US dollars, prediabetes merged with
  diabetes. These are stated in the HLD and the user manual.
* **Split:** stratified 70 / 15 / 15 (177,576 / 38,052 / 38,052 rows).

## 3. System overview

The full architecture diagram with an explanation of every block is in
[docs/architecture.md](docs/architecture.md). In short:

* **Offline pipeline (DVC DAG):** `ingest -> process (Ray) -> train (MLflow) -> evaluate (acceptance gate + registry)`.
* **Online services (Docker Compose, 8 containers):** frontend (nginx), backend (FastAPI), model server
  (`mlflow models serve`), MLflow tracking server + registry, Prometheus, Alertmanager, Grafana, node-exporter.
* **Loose coupling:** the UI talks to the backend only through a configurable REST URL (`API_BASE_URL`), and the
  backend talks to the model server only through REST (`/invocations`). Each block can be replaced independently.
* **Monitoring loop:** backend metrics -> Prometheus -> Grafana dashboard and alert rules; live drift (PSI);
  a drift monitor that triggers retraining; a rollback script.

## 4. Lifecycle stages and how I implemented them

| Lifecycle stage (development guideline) | Implementation |
|---|---|
| A. Problem definition and data collection | success metrics above; automated download (`src/data/ingest.py`) with schema check and MD5 of the raw file |
| B. Preprocessing and feature engineering | Ray Data validation, quality gate and cleaning (`src/data/process.py`); feature engineering inside the sklearn Pipeline (`src/common/features.py`) so training and serving use the same code |
| C. Model development and training | 6 candidates (LR, RF, XGBoost x 2 settings each), class weights, threshold tuned for F2, all tracked in MLflow (`src/models/train.py`) |
| D. Deployment | acceptance gate + MLflow Model Registry `@champion`; MLflow model server behind a FastAPI backend; Docker Compose; rollback (`scripts/rollback.sh`) |
| E. Monitoring and maintenance | Prometheus metrics for every component, 7 alert rules, Grafana dashboard, live PSI drift, drift-triggered retraining (`src/monitoring/drift_monitor.py`) |

## 5. Model development and results

### 5.1 Approach

* **Class imbalance:** class weights (`class_weight="balanced"`, XGBoost `scale_pos_weight`) instead of
  resampling, so the training distribution stays the real one (which drift monitoring compares against).
* **Decision threshold:** chosen on the validation split to maximise F-beta with beta = 2 ("recall counts twice
  as much as precision"), instead of the default 0.5.
* **Selection:** the candidate with the highest validation ROC-AUC is selected automatically; the test split is
  used only once, by the evaluate stage.

### 5.2 Model comparison (validation split)

| Candidate | ROC-AUC | Recall | Precision | F1 | Train time |
|---|---|---|---|---|---|
| logistic_regression (C=0.1) | 0.8256 | 0.828 | 0.289 | 0.428 | 2.5 s |
| logistic_regression (C=1.0) | 0.8256 | 0.828 | 0.289 | 0.428 | 3.0 s |
| random_forest (100 trees, depth 10) | 0.8267 | 0.846 | 0.283 | 0.424 | 5.0 s |
| random_forest (150 trees, depth 12) | 0.8268 | 0.833 | 0.287 | 0.427 | 7.7 s |
| **xgboost (300 trees, depth 4) - selected** | **0.8299** | 0.807 | 0.304 | 0.442 | 1.0 s |
| xgboost (500 trees, depth 6) | 0.8271 | 0.823 | 0.292 | 0.431 | 2.3 s |

All candidates are within 0.005 ROC-AUC of each other, which tells me the limit is the information in 21
self-reported answers rather than the algorithm.

![Model comparison and drift in the pipeline console](screenshots/09_console_models_compared_drift.png)

### 5.3 Held-out test set (38,052 people)

ROC-AUC **0.834**, recall **0.818**, precision **0.301**, F1 0.440, PR-AUC 0.440, accuracy 0.710, threshold 0.480.
Of 5,302 people with diabetes or prediabetes, the model flags 4,335 and misses 967, at the cost of 10,063 false
alarms. For a screening tool whose follow-up is a cheap blood test, this trade-off is intended. Accuracy is not a
target: a model that always answers "no diabetes" would score 0.86 and be useless.

### 5.4 Explainability

For each person, the backend re-scores "what-if" versions of their answers in which one changeable risk factor
(for example BMI, physical activity, general health) is set to a healthier value, and shows the three changes that
lower the score most. Things a person cannot change (age, sex) are never suggested. This is model-agnostic, costs
one batched model call, and gives advice a non-technical user can act on. Global feature importance is logged to
MLflow for every selected model.

## 6. AIOps implementation

### 6.1 Data engineering (Ray)

`src/data/process.py` reads the CSV into a Ray Dataset split into 8 blocks processed on 4 CPUs: per-block quality
statistics (nulls, ranges, non-integer codes, labels), a quality gate that stops the pipeline if the data is
implausible, and parallel cleaning. Measured throughput on my VM: **22,723 rows/s** for the Ray processing and
14,194 rows/s for the whole stage including Ray start-up (17.9 s for 253,680 rows). Stage durations of the full
pipeline: ingest 19.1 s (download), process 17.9 s, train 50.6 s, evaluate 2.2 s.

![Ray processing results and test-set metrics in the pipeline console](screenshots/08_console_data_and_model.png)

### 6.2 Source control and continuous integration

* **Git** for code, configuration, metrics and documents; **Git LFS** for the screenshots; **DVC** for the raw
  and processed data and the models (local DVC remote, no cloud).
* **DVC DAG** (`dvc.yaml`, shown with `dvc dag`):

  ```
  ingest -> process -> train -> evaluate
               \___________________^
  ```

  Each stage declares its code, data and parameter dependencies, so `dvc repro` re-runs only what changed. I saw
  this in practice: after I changed `evaluate.py`, CI re-ran only the evaluate stage.
* **CI gate** (`scripts/ci.sh`): flake8 -> 74 tests + test report -> `dvc repro` (including the acceptance gate)
  -> `dvc metrics diff` against the last commit. It runs automatically as a Git pre-push hook.

| DVC stages with their last run | Every pipeline run, successful or failed |
|---|---|
| ![DVC pipeline in the console](screenshots/07_console_dvc_pipeline.png) | ![Run history](screenshots/10_console_run_history.png) |

### 6.3 Experiment tracking (MLflow)

Every training run is one parent run with one nested run per candidate. I did not rely on autolog; everything
is logged explicitly:

* **Parameters:** model type and hyperparameters, seed, selection metric, threshold beta; `params.yaml` and the
  data-quality report as artifacts.
* **Metrics:** validation accuracy, precision, recall, F1, ROC-AUC, PR-AUC, threshold, training time and p95
  scoring latency per candidate; test metrics on the selected run.
* **Artifacts:** confusion matrix and ROC plots (validation and test), feature-importance plot and JSON, model
  comparison, acceptance result, and the MLflow pyfunc model with signature and input example.
* **Tags for reproducibility:** `git_commit`, `git_dirty`, `train_data_md5`, `model_type`, `best_model`,
  `acceptance`. Every registered model version also carries its `git_commit` and test scores.

| Model registry: `@champion` | Every version with its Git commit and test scores |
|---|---|
| ![MLflow registry](screenshots/18_mlflow_registry.png) | ![MLflow model versions](screenshots/19_mlflow_model_versions.png) |

Version 1 was trained before I fixed the row-order problem (section 12) and scored 0.8300; versions 2 and 3,
trained from the same commit after the fix, are identical (0.8341).

### 6.4 Exporter instrumentation and visualisation (Prometheus, Grafana)

The backend exposes `/metrics` with request counts and latency per endpoint, in-progress requests, handled
exceptions by type, predictions by risk level, the risk-score histogram, inference latency, the served model and
its test metrics, PSI per feature, drift status, the state, duration and throughput of every pipeline stage, and
the health of every other service (`dependency_up`). node-exporter adds CPU and memory. Prometheus scrapes every
5 s and evaluates **7 alert rules** (backend down, model server down, error rate > 5 %, p95 > 500 ms, data drift,
pipeline stage failed, high-risk share spike), routed to Alertmanager. Grafana loads a provisioned dashboard
with **20 panels in 6 rows** that refreshes every 5 s (near real time).

| Service health and API traffic | Model outputs and drift per feature |
|---|---|
| ![Grafana service health and API](screenshots/12_grafana_service_health_api.png) | ![Grafana model and drift](screenshots/13_grafana_model_and_drift.png) |
| **Pipeline stages and infrastructure** | **Prometheus: every scrape target up** |
| ![Grafana pipeline and infrastructure](screenshots/15_grafana_pipeline_and_infra.png) | ![Prometheus targets](screenshots/16_prometheus_targets.png) |

In the first Grafana screenshot, the 503 responses and `model_unavailable` exceptions around 19:37 are real:
my VM was overloaded at that moment, the model server stopped answering, and the backend returned the friendly
503 instead of crashing.

### 6.5 Software packaging

* **MLflow for APIfication:** the model is saved as an MLflow pyfunc model (with the feature-engineering code
  bundled) and served by `mlflow models serve` as the inference engine.
* **MLflow Project:** `MLproject` with `python_env.yaml` defines the entry points `main`, `ingest`, `process`,
  `train`, `evaluate`, `drift`, `drift_retrain` and `test`; I ran `mlflow run . -e test` successfully (74 passed).
* **FastAPI** exposes the REST API the UI needs (predict, model card, pipeline console, drift, health, metrics);
  its OpenAPI page is at `/docs`.
* **Docker:** separate images for the frontend, the backend and MLflow; **Docker Compose** runs them as separate
  services with health checks and start-up ordering.

## 7. Software engineering

* **Design documents:** [architecture](docs/architecture.md), [HLD](docs/HLD.md) (design choices and rationale),
  [LLD](docs/LLD.md) (every endpoint with its input/output specification, error codes, modules, configuration,
  metrics).
* **Paradigm:** object-oriented where there is state or interchangeable behaviour (`BasePredictor` with
  `LocalPredictor` / `MlflowServerPredictor` as a strategy chosen by a factory, `PredictionStore`, `Settings`),
  functional elsewhere (pipeline stages, metrics and drift maths as pure, testable functions).
* **Coding style:** PEP 8 with a 120-character line limit, checked with flake8 in CI (configuration in
  `setup.cfg`); docstrings and inline comments throughout.
* **Logging:** one consistent log format for every pipeline stage and the backend; every API request is logged
  with method, path, status, duration and a request ID that is also returned to the client.
* **Exception handling:** custom exceptions (`DataQualityError`, `AcceptanceError`, `PredictorError`); the API
  maps errors to 422 (with the invalid field names), 503 (model unavailable, friendly message) and 500 (no internal
  details exposed); pipeline failures are recorded and raised as alerts.
* **Security:** inputs are logged only in encrypted form (Fernet), no personal identifiers are collected, the
  backend runs as a non-root user, CORS is restricted to the UI origin, and model reload needs an admin token.

## 8. Demonstration

* **Web UI** ([user manual](docs/user_manual.md), also built into the app under *Help*): a six-step
  questionnaire with large Yes/No buttons, plain-language questions, a BMI calculator, validation that highlights
  missing or out-of-range answers, a result page with a colour-coded risk level, a score bar, the top three
  changeable factors and a disclaimer. It works on phone-sized screens and with the keyboard only. The font,
  Atkinson Hyperlegible, was designed for low-vision readers.
* **ML pipeline visualisation and management console:** the app's *How it works* page shows the DVC DAG with the
  status of each stage, run history with successes and failures and their errors, data-quality checks, Ray
  throughput, the model comparison, test metrics with the acceptance gate, live drift per feature, and links to
  MLflow, Grafana, Prometheus and Alertmanager.

| Step 1 of the questionnaire | Yes / No questions |
|---|---|
| ![App home](screenshots/01_app_home.png) | ![Health conditions](screenshots/02_app_form_health_conditions.png) |
| **Result: level, score and what to change** | **Advice and disclaimer** |
| ![Result](screenshots/05_app_result.png) | ![Advice](screenshots/06_app_result_advice.png) |
| **Number answers with their allowed range** | **Pipeline console: run history and tool links** |
| ![Questions with numbers](screenshots/03_app_form_feeling.png) | ![Console tools](screenshots/11_console_tools.png) |

All screenshots: [screenshots/](screenshots/).

## 9. Testing

* **Test plan, test cases and acceptance criteria:** [docs/test_plan.md](docs/test_plan.md).
* **Automated:** 74 tests (unit and integration: data validation, the Ray pipeline, training utilities, the
  acceptance gate, drift, encryption, every API endpoint and error path, registration retries): **74 passed,
  0 failed** ([docs/test_report.md](docs/test_report.md)).
* **System tests (S1-S9):** all passed, including a real model-server outage, drift, load test, retraining and
  rollback. **Acceptance criteria AC1-AC8:** all met, including 3 of 3 non-technical testers completing the
  questionnaire without help.

## 10. Monitoring, drift, retraining and rollback in practice

| What I did | What happened |
|---|---|
| Sent 500 normal requests | no drift (max PSI 0.020) |
| Sent 400 shifted requests (older, heavier, less active people) | drift detected on BMI, PhysActivity and Age (max PSI 1.77); `DataDriftDetected` alert fired |
| Stopped the model server | users got HTTP 503 with "The prediction service is temporarily unavailable"; `ModelServerDown` alert fired; restart recovered |
| Ran `drift_monitor --retrain` | drift confirmed from the encrypted log; `dvc repro --force` retrained (about 3 min); the new model passed the gate and became version 4 `@champion` |
| Ran `scripts/rollback.sh HEAD~1` | the previous model was restored from the DVC cache in seconds, `@champion` moved back to version 3, and the app served it again |

Load test (10 s per level): 1 user 30 requests/s with p95 44 ms; 4 users p95 232 ms; 8 users p95 517 ms, with 0
errors. At 8 users the load generator and all 8 containers compete for the same 6 vCPUs.

## 11. Reproducibility

Every experiment can be reproduced from a Git commit and its MLflow run ID: the run carries the `git_commit`
tag, `dvc.lock` pins the exact data and model hashes of that commit, and seeds are fixed in `params.yaml`
(`git checkout <commit> && dvc checkout`, or `dvc repro`). I checked this directly: two complete forced re-runs
from the same commit, and a later drift-triggered retraining, all produced identical metrics (test ROC-AUC 0.8341).

## 12. Problems I found and fixed

| Problem | Cause | Fix |
|---|---|---|
| The same seed gave different scores on two runs | Ray returned rows in a different order on each run, so the seeded split picked different people | sort the cleaned rows into a canonical order before the split; added a test |
| A clear population shift did not raise a drift alert | the drift window of 5,000 requests diluted 400 shifted requests | window of the 500 most recent predictions, configurable in `params.yaml` |
| Retraining failed when registering the model | under heavy CPU load, the only MLflow server worker was restarted mid-request | 2 MLflow workers; registration retried and made idempotent; added a test |
| The backend could not find the model after retraining | DVC recreates `models/`, and a Docker bind mount keeps pointing at the deleted folder | restart the model server and backend after retraining (the rollback script does this) |
| Grafana showed healthy values in red and "No data" for errors | default thresholds; an empty error series | neutral colours for informational panels; report 0 when there are no errors |
| Containers could not read files on my VM's shared folder | the VirtualBox share was mounted root-only | mount it with open permissions (documented in the README) |

## 13. Limitations and future work

* The score ranks risk; it is not a calibrated probability, and the UI says so. Calibration (e.g. isotonic
  regression on the validation split) would allow a true percentage.
* The data is a 2015 US survey; the model may transfer imperfectly to other populations and today's population.
* Retraining re-uses the public dataset; a real deployment would add newly labelled data before retraining.
* Everything runs on one machine with Docker Compose; the same images could be deployed on Kubernetes or
  Docker Swarm for scaling.

## 14. Coverage of the evaluation guideline

| Guideline item | Where it is covered |
|---|---|
| UI/UX, foolproof input, responsiveness, user manual | Section 8; [user manual](docs/user_manual.md); test plan 8.4 |
| ML pipeline visualisation, management console, errors/failures/successes, pipeline speed | Section 8 (*How it works* page); Grafana pipeline row; Sections 6.1, 6.4 |
| Design document, paradigm, LLD with endpoints and I/O, architecture and HLD diagrams, loose coupling | Section 7; [architecture](docs/architecture.md), [HLD](docs/HLD.md), [LLD](docs/LLD.md) |
| Coding style, logging, exception handling, inline documentation, unit tests | Section 7; `setup.cfg`; `tests/` |
| Test plan, test cases, test report, acceptance criteria and their result | Section 9; [test plan](docs/test_plan.md); [test report](docs/test_report.md) |
| Data engineering with Ray, throughput | Section 6.1 |
| CI with DVC, DVC DAG, Git / Git LFS / DVC versioning | Section 6.2 |
| Experiment tracking: metrics, parameters, artifacts, beyond autolog | Section 6.3 |
| Prometheus instrumentation, monitored components, Grafana in near real time | Section 6.4 |
| MLflow APIfication, MLproject, FastAPI, Dockerised frontend and backend, Docker Compose | Section 6.5 |
| Reproducibility via Git commit + MLflow run ID | Section 11 |

## 15. How to run it

Step-by-step instructions, from a fresh clone to the running app and the monitoring demo, are in
[README.md](README.md).
