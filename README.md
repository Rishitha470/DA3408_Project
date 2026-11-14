# Diabetes Risk Prediction System (AIOps)

A web application that estimates a person's diabetes risk from 21 simple health questions, built
end to end with AIOps practices: Ray data pipeline, DVC pipeline/CI, MLflow tracking + registry +
serving, FastAPI backend, separate web frontend, Docker Compose, Prometheus/Grafana monitoring,
drift detection with retraining, and rollback.

> Screening and awareness only. Not a medical diagnosis.

| Document | Path |
|---|---|
| Architecture diagram + block explanations | [docs/architecture.md](docs/architecture.md) |
| High-level design (choices and rationale) | [docs/HLD.md](docs/HLD.md) |
| Low-level design (API endpoints, I/O spec) | [docs/LLD.md](docs/LLD.md) |
| Test plan and test cases | [docs/test_plan.md](docs/test_plan.md) |
| Test report (generated) | [docs/test_report.md](docs/test_report.md) |
| User manual (non-technical) | [docs/user_manual.md](docs/user_manual.md) |

## Repository layout

```
src/                  pipeline code (shared by training and serving)
  common/             schema, feature engineering, drift (PSI), encryption, logging, run tracking
  data/               ingest.py (stage 1), process.py (stage 2, Ray), synthetic.py (tests only)
  models/             train.py (stage 3, MLflow), evaluate.py (stage 4, registry), pyfunc.py
  monitoring/         drift_monitor.py (drift check + retraining trigger)
backend/              FastAPI REST API (Dockerfile, app/)
frontend/             web UI: html/ (index, pipeline console, help), nginx, Dockerfile
docker/               MLflow image, pipeline image, model requirements
monitoring/           Prometheus config + alert rules, Alertmanager, Grafana provisioning + dashboard
tests/                73 automated tests (pytest)
scripts/              run_tests, ci, load_test, simulate_traffic, rollback, build_dashboard
docs/                 the mandatory documents
dvc.yaml params.yaml  DVC pipeline (DAG) and all parameters
MLproject python_env.yaml   MLflow Project (identical dev/test environments)
docker-compose.yml    all runtime services
```

## Prerequisites

* Python **3.12** (3.11 also works), Git, Git LFS (`git lfs install`)
* Docker Desktop (or Docker Engine + Compose v2), about 8 GB RAM free
* Internet access for the first run (dataset download, Docker images)
* Windows: use **WSL2** (Ubuntu) for the commands below. Native PowerShell works too, but replace
  `export X=y` with `$env:X="y"` and `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`.

## Step-by-step: from zero to running app

### 1. Set up the environment

```bash
cd diabetes-risk-aiops
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git init
git lfs install
dvc init
mkdir -p ../dvc-storage && dvc remote add -d localstore ../dvc-storage   # local DVC remote (no cloud)
git add -A && git commit -m "Initial project structure"
```

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

This takes roughly 5-15 minutes depending on your laptop. If the evaluate stage stops with
"model rejected", the acceptance criteria in `params.yaml` were not met (see `reports/acceptance.json`).
If the UCI download fails, download `diabetes_binary_health_indicators_BRFSS2015.csv` (Kaggle, "Diabetes Health
Indicators Dataset") into `data/external/` and set `data.source: csv` in `params.yaml`.

### 5. Run the tests and generate the test report

```bash
python scripts/run_tests.py        # 73 tests -> docs/test_report.md (includes acceptance results)
```

### 6. Commit the versioned results

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
docker compose restart model-server                 # serve the new model
curl -X POST localhost:8000/api/v1/model/reload -H "X-Admin-Token: <ADMIN_TOKEN from .env>"

./scripts/rollback.sh HEAD~1                        # restore the previous model from DVC and restart
```

## Continuous integration

```bash
./scripts/ci.sh                    # lint -> tests + report -> dvc repro -> metrics diff vs. last commit
./scripts/install_git_hook.sh      # run ci.sh automatically before every git push
dvc exp run --set-param train.threshold_beta=1.5     # try a parameter change as a DVC experiment
dvc exp show
```

## MLflow Project (identical dev/test environments)

```bash
mlflow run . -e train --env-manager local --experiment-name diabetes-risk
mlflow run . -e test  --env-manager local --experiment-name diabetes-risk
mlflow run . --env-manager virtualenv --experiment-name diabetes-risk    # fresh env from python_env.yaml
```

Entry points: `main` (whole DVC pipeline), `ingest`, `process`, `train`, `evaluate`, `drift`, `drift_retrain`, `test`.

## Running the pipeline inside Docker (optional)

```bash
docker compose --profile pipeline run --rm pipeline       # runs `dvc repro` in a container, logs to MLflow
```

## Running without Docker (development)

```bash
uvicorn backend.app.main:app --port 8000      # INFERENCE_MODE=local: loads models/model.joblib
cd frontend/html && python -m http.server 8080
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
