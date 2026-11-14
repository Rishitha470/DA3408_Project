# Diabetes Risk Prediction System

A web app that tells you how your diabetes risk looks based on 21 everyday health questions: age, weight,
blood pressure, how active you are, and so on. It also tells you which changes would bring your risk down the
most.

Behind the app there's a full pipeline that I set up and ran myself. Ray cleans and checks the data, DVC
versions the data and models and re-runs only what changed, MLflow tracks every experiment and serves the
model, and FastAPI sits between the model and the web page. Everything runs in Docker Compose. Prometheus and
Grafana watch it all, and when the incoming data starts to look different from the training data, the model
can be retrained, or rolled back if something goes wrong.

> This is a screening and awareness tool, not a medical diagnosis.

If you want the whole story (design choices, results, what went wrong and how I fixed it), read
[REPORT.md](REPORT.md). How I used AI tools is in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).

## Results at a glance

From my run on the CDC Diabetes Health Indicators data (253,680 survey answers); details in
[docs/HLD.md section 8](docs/HLD.md#8-results).

| | Result |
|---|---|
| Selected model | XGBoost (300 trees, depth 4), best of 6 candidates (LR, RF, XGBoost) by validation ROC-AUC |
| Held-out test set (38,052 people) | ROC-AUC **0.834**, recall **0.818**, precision 0.301, F1 0.440; all acceptance criteria passed |
| Ray data pipeline | 253,680 rows validated and cleaned at 22,723 rows/s (4 CPUs, 8 blocks) |
| API latency | p95 44 ms for 1 user, 232 ms for 4 concurrent users (target < 500 ms) |
| Tests | 74 automated tests, 74 passed ([docs/test_report.md](docs/test_report.md)) |
| Drift | shifted traffic detected on BMI, PhysActivity, Age (PSI 1.77); alert fires; the model is retrained and can be rolled back |
| Reproducibility | two full re-runs from the same commit gave identical metrics; every model version is tagged with its Git commit |

## Screenshots

| The app: result page | Pipeline console ("How it works" page) |
|---|---|
| ![Result page](screenshots/05_app_result.png) | ![Pipeline console](screenshots/07_console_dvc_pipeline.png) |
| **Grafana: model and drift** | **MLflow: registered model versions** |
| ![Grafana](screenshots/13_grafana_model_and_drift.png) | ![MLflow registry](screenshots/19_mlflow_model_versions.png) |

All 19 screenshots (app, pipeline console, Grafana, Prometheus, MLflow) are in [screenshots/](screenshots/).

## Documents

| Document | Path |
|---|---|
| Project report (summary, rubric coverage, results) | [REPORT.md](REPORT.md) |
| AI use disclosure | [AI_DISCLOSURE.md](AI_DISCLOSURE.md) |
| Architecture diagram + block explanations | [docs/architecture.md](docs/architecture.md) |
| High-level design (choices and rationale) | [docs/HLD.md](docs/HLD.md) |
| Low-level design (API endpoints, I/O spec) | [docs/LLD.md](docs/LLD.md) |
| Test plan and test cases | [docs/test_plan.md](docs/test_plan.md) |
| Test report (generated) | [docs/test_report.md](docs/test_report.md) |
| User manual (non-technical) | [docs/user_manual.md](docs/user_manual.md) |

## Repository layout

```
.
├── README.md                     you are here: what it is, how to run it
├── REPORT.md                     project report: design, results, problems fixed, rubric coverage
├── AI_DISCLOSURE.md              how I used AI tools
│
├── docs/                         the mandatory documents
│   ├── architecture.md           architecture diagram + what each block does + deployment table
│   ├── HLD.md                    high-level design: choices and rationale, results (section 8)
│   ├── LLD.md                    low-level design: every API endpoint with input/output, modules, config
│   ├── test_plan.md              test plan, 58 test cases, acceptance criteria, execution results
│   ├── test_report.md            generated test report: 74 tests, passed / failed
│   └── user_manual.md            user manual for non-technical users (also in the app under Help)
├── screenshots/                  19 screenshots: app, pipeline console, Grafana, Prometheus, MLflow
│
├── dvc.yaml                      DVC pipeline (DAG): ingest -> process -> train -> evaluate
├── dvc.lock                      exact hashes of the data, models and code of the last run
├── params.yaml                   every tunable setting (data, Ray, models, threshold, acceptance, drift)
├── MLproject, python_env.yaml    MLflow Project entry points (identical dev/test environments)
├── docker-compose.yml            all 8 runtime services
├── requirements.txt              Python dependencies (pinned)
│
├── src/                          pipeline code (shared by training and serving)
│   ├── data/
│   │   ├── ingest.py             stage 1: download the CDC data, check the schema
│   │   ├── process.py            stage 2: Ray validation, quality gate, cleaning, reproducible split
│   │   └── synthetic.py          small synthetic dataset for the tests
│   ├── models/
│   │   ├── train.py              stage 3: 6 candidates, threshold tuning, MLflow tracking
│   │   ├── evaluate.py           stage 4: test set, acceptance gate, MLflow registry (@champion)
│   │   └── pyfunc.py             the MLflow model that the model server runs
│   ├── monitoring/
│   │   └── drift_monitor.py      batch drift check, triggers retraining
│   └── common/                   schema, feature engineering, PSI drift, encryption, logging, run log
│
├── backend/                      FastAPI REST API (Dockerfile + app/)
│   └── app/
│       ├── main.py               endpoints, middleware, error handling
│       ├── predictor.py          calls the MLflow model server (or loads the model locally)
│       ├── explain.py            "what would lower your risk most"
│       ├── prediction_store.py   encrypted input log + live drift window
│       ├── metrics.py            Prometheus metrics
│       ├── pipeline_status.py    data for the pipeline console
│       ├── schemas.py            request/response validation
│       └── config.py             settings from environment variables
├── frontend/                     web UI served by nginx (Dockerfile)
│   └── html/                     index.html (risk check), pipeline.html (console), manual.html (help), css/, js/
├── docker/                       MLflow image (tracking server + model server), model requirements
├── monitoring/
│   ├── prometheus/               scrape config + 7 alert rules
│   ├── alertmanager/             alert routing
│   └── grafana/                  provisioned data source + dashboard (20 panels)
│
├── reports/                      results of the last run (tracked by DVC as metrics/plots)
│   ├── ingest.json, data_quality.json       data statistics and Ray throughput
│   ├── model_comparison.json                all 6 candidates on the validation split
│   ├── metrics.json, acceptance.json        test-set metrics and the acceptance gate
│   └── plots/                               ROC, precision-recall and confusion-matrix data
├── tests/                        74 automated tests (pytest)
└── scripts/
    ├── ci.sh                     CI gate: lint -> tests -> dvc repro -> metrics diff
    ├── install_git_hook.sh       runs ci.sh before every push (and uploads the Git LFS files)
    ├── run_tests.py              runs the tests and writes docs/test_report.md
    ├── simulate_traffic.py       normal or drifted traffic for the monitoring demo
    ├── load_test.py              throughput and latency at different numbers of users
    ├── rollback.sh               return to the model of an earlier commit
    └── build_dashboard.py        generates the Grafana dashboard
```

Not in the repository (created by running it): `data/` and `models/` (versioned with DVC), `logs/`, and `.env`
(your secrets, see step 2).

## Prerequisites

* Linux with Python **3.12**, Git, Git LFS and Docker Engine + Compose v2. On Windows, use WSL2 (Ubuntu) or a
  Linux VM.
* 8 GB RAM recommended (I used a 7.2 GB VM; close other programs while training)
* Internet access for the first run (dataset download, Docker images)

**My test environment:** VirtualBox VM, Ubuntu 26.04, 6 vCPUs, 7.2 GB RAM, Python 3.12.16 (via `uv`),
Docker 29.1 / Compose 2.40, DVC 3.67.1, MLflow 3.18.0, Ray 2.60.0.

## Step-by-step: from zero to running app

### 1. Set up the environment

```bash
git clone <repository-url> diabetes-risk-aiops && cd diabetes-risk-aiops
git lfs install && git lfs pull          # screenshots are stored with Git LFS
python3.12 -m venv .venv                 # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
```

Git, Git LFS and DVC are already initialised in the repository. The DVC remote is a local folder
(`../dvc-storage`, no cloud). If you do not have that folder, step 4 simply rebuilds the data and models.

**VirtualBox shared folder:** if the project lives on a VirtualBox shared folder, mount it so that container
users can read it, e.g. `sudo mount -t vboxsf -o uid=1000,gid=1000,dmode=777,fmode=777 <share> /mnt/shared`,
and create the virtual environment outside the shared folder (for example in your home directory).

### 2. Create the secrets file

```bash
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste the printed key after PREDICTION_LOG_KEY= in .env, and set ADMIN_TOKEN to any password
```

### 3. Start the MLflow tracking server

```bash
docker compose up -d --build mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000      # every pipeline command below logs here
```

Open http://localhost:5000 to check it is up. (Without Docker you can skip this step: the code then logs to a
local `mlflow.db`, viewable with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.)

### 4. Run the pipeline (downloads the real CDC data, ~250k rows)

```bash
dvc repro          # ingest -> process (Ray) -> train (6 MLflow runs) -> evaluate (+ registry)
dvc dag            # show the pipeline DAG
dvc metrics show   # data quality, model comparison, test metrics, acceptance
dvc plots show     # writes dvc_plots/index.html (ROC, precision-recall, confusion matrix)
```

On my VM this takes about 3 minutes (the first run also downloads the dataset). If the evaluate stage stops with
"model rejected", the acceptance criteria in `params.yaml` were not met (see `reports/acceptance.json`).
If the UCI download fails, download `diabetes_binary_health_indicators_BRFSS2015.csv` (Kaggle, "Diabetes Health
Indicators Dataset") into `data/external/` and set `data.source: csv` in `params.yaml`.

### 5. Run the tests and generate the test report

```bash
python scripts/run_tests.py        # 74 tests -> docs/test_report.md (includes acceptance results)
```

### 6. Commit the versioned results (after every training run)

```bash
git add dvc.lock reports/*.json reports/plots docs/test_report.md
git commit -m "Train model on CDC data"
dvc push                           # store data and models in the local DVC remote
```

Every MLflow run is tagged with this Git commit (`git_commit`) and the training data hash (`train_data_md5`).

### 7. Start the whole application

```bash
mkdir -p logs && chmod 777 logs    # the backend container writes the encrypted prediction log here
docker compose up -d --build
docker compose ps                  # wait until backend and model-server show "healthy"
```

| Service | URL | Notes |
|---|---|---|
| Web app (frontend) | http://localhost:8080 | risk check, "How it works" pipeline console, Help |
| Backend API docs | http://localhost:8000/docs | interactive OpenAPI |
| MLflow (experiments, registry) | http://localhost:5000 | experiment `diabetes-risk`, model `diabetes-risk-model@champion` |
| Model server | http://localhost:5001/ping | MLflow model serving |
| Grafana | http://localhost:3000 | dashboard "Diabetes Risk - AIOps monitoring" (admin / admin) |
| Prometheus | http://localhost:9090 | Status -> Targets, Alerts |
| Alertmanager | http://localhost:9093 | fired alerts |

### 8. Demo the monitoring

```bash
python scripts/simulate_traffic.py --n 300            # normal traffic (real test rows) -> no drift
python scripts/simulate_traffic.py --n 400 --drift    # older/heavier population -> drift alert
python scripts/load_test.py --concurrency 1 4 8 --duration 10      # throughput and latency table
docker compose stop model-server    # app shows a friendly message, Grafana shows DOWN, alert fires
docker compose start model-server
```

### 9. Drift-triggered retraining, reload and rollback

```bash
set -a; source .env; set +a                         # the drift monitor needs PREDICTION_LOG_KEY to decrypt
python -m src.monitoring.drift_monitor              # writes reports/drift.json
python -m src.monitoring.drift_monitor --retrain    # runs `dvc repro --force` if drift is detected
docker compose restart model-server backend         # serve the new model (see note below)

./scripts/rollback.sh HEAD~1                        # restore the previous model from DVC and restart
```

Why restart and not just reload? `dvc repro` deletes and recreates `models/`, and a Docker bind mount keeps
pointing at the deleted folder. I found this out when the backend couldn't see the new model after retraining.
Restarting the containers fixes it, and `rollback.sh` does the same.

## Continuous integration

```bash
./scripts/ci.sh                    # lint -> tests + report -> dvc repro -> metrics diff vs. last commit
./scripts/install_git_hook.sh      # run ci.sh automatically before every git push
```

When I changed `evaluate.py`, CI re-ran only the evaluate stage and the metrics diff came back empty, so the
change didn't affect the model.

## MLflow Project

`MLproject` (with `python_env.yaml`) gives every step of the project a fixed entry point, so development and
testing run the same way: `main` (whole DVC pipeline), `ingest`, `process`, `train`, `evaluate`, `drift`,
`drift_retrain` and `test`. For example, the test suite as a tracked MLflow run:

```bash
mlflow run . -e test --env-manager local --experiment-name diabetes-risk
```

## Reproducing a result

```bash
git checkout <commit>      # the git_commit tag of the MLflow run
dvc checkout               # restores the exact data and model of that commit
dvc repro                  # or re-run; unchanged stages are skipped
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `PREDICTION_LOG_KEY` error on `docker compose up` | Step 2: create `.env` with a key |
| model-server unhealthy | `models/` missing: run step 4 first; check `docker compose logs model-server` |
| Web app says "could not reach the risk service" | `docker compose ps` backend healthy? Open the app via http://localhost:8080 |
| Ray errors about shared memory | Give Docker/WSL more memory, or lower `process.num_cpus` in `params.yaml` |
| Port already in use | Stop the other program or change the left side of `ports:` in `docker-compose.yml` |
| Backend log: "could not write prediction log" | `chmod 777 logs` (container user cannot write the folder) |
| `dvc repro` says nothing to do | Nothing changed; use `dvc repro --force` to re-run everything |
| Training runs log to a local `mlflow.db` instead of the server | `MLFLOW_TRACKING_URI` is not set in this terminal: `export MLFLOW_TRACKING_URI=http://localhost:5000` |
| Prometheus/Grafana "permission denied" on a VirtualBox shared folder | mount the share with `dmode=777,fmode=777` (step 1) |
| Desktop freezes during training | not enough RAM: close the browser and other programs while `dvc repro` runs |
