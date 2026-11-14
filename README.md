# Diabetes Risk Prediction System (AIOps)

A web application that estimates a person's diabetes risk from 21 simple health questions, built
end to end with AIOps practices: Ray data pipeline, DVC pipeline/CI, MLflow tracking + registry +
serving, FastAPI backend, separate web frontend, Docker Compose, Prometheus/Grafana monitoring,
drift detection with retraining, and rollback.

> Screening and awareness only. Not a medical diagnosis.

## Results at a glance

Real run on the CDC Diabetes Health Indicators data (253,680 survey answers); details in
[docs/HLD.md section 8](docs/HLD.md#8-results).

| | Result |
|---|---|
| Selected model | XGBoost (300 trees, depth 4), best of 6 candidates (LR, RF, XGBoost) by validation ROC-AUC |
| Held-out test set (38,052 people) | ROC-AUC **0.834**, recall **0.818**, precision 0.301, F1 0.440; all acceptance criteria passed |
| Ray data pipeline | 253,680 rows validated and cleaned at 22,723 rows/s (4 CPUs, 8 blocks) |
| API latency | p95 44 ms for 1 user, 232 ms for 4 concurrent users (target < 500 ms) |
| Tests | 74 automated tests, 74 passed ([docs/test_report.md](docs/test_report.md)) |
| Drift | shifted traffic detected on BMI, PhysActivity, Age (PSI 1.77); alert fires; retraining and rollback verified |
| Reproducibility | two full re-runs from the same commit gave identical metrics; every model version is tagged with its Git commit |

Screenshots of the app, pipeline console, Grafana, Prometheus and MLflow are in [docs/screenshots/](docs/screenshots/).

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
tests/                74 automated tests (pytest)
scripts/              run_tests, ci, load_test, simulate_traffic, rollback, build_dashboard
docs/                 the mandatory documents
dvc.yaml params.yaml  DVC pipeline (DAG) and all parameters
MLproject python_env.yaml   MLflow Project (identical dev/test environments)
docker-compose.yml    all runtime services
```

## Prerequisites

* Linux with Python **3.12**, Git, Git LFS and Docker Engine + Compose v2. On Windows, use WSL2 (Ubuntu) or a
  Linux VM.
* 8 GB RAM recommended (tested on a 7.2 GB VM: close other programs while training)
* Internet access for the first run (dataset download, Docker images)

**Tested environment:** VirtualBox VM, Ubuntu 26.04, 6 vCPUs, 7.2 GB RAM, Python 3.12.16 (via `uv`),
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

This takes roughly 5-15 minutes depending on your laptop. If the evaluate stage stops with
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

Note: `dvc repro` deletes and recreates `models/`. A Docker bind mount keeps pointing at the deleted folder,
so containers must be **restarted** (not just asked to reload) after retraining; `rollback.sh` does this too.
`POST /api/v1/model/reload` (with header `X-Admin-Token`) is for the backend run without Docker (`uvicorn`).

## Continuous integration

```bash
./scripts/ci.sh                    # lint -> tests + report -> dvc repro -> metrics diff vs. last commit
./scripts/install_git_hook.sh      # run ci.sh automatically before every git push
dvc exp run --set-param train.threshold_beta=1.5     # try a parameter change as a DVC experiment
dvc exp show
```

Note: an experiment that passes the acceptance gate is also registered in MLflow and becomes `@champion`;
use `scripts/rollback.sh <commit>` to return to the committed model afterwards.

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
| Training runs log to a local `mlflow.db` instead of the server | `MLFLOW_TRACKING_URI` is not set in this terminal: `export MLFLOW_TRACKING_URI=http://localhost:5000` |
| Prometheus/Grafana "permission denied" on a VirtualBox shared folder | mount the share with `dmode=777,fmode=777` (step 1) |
| Desktop freezes during training | not enough RAM: close the browser and other programs while `dvc repro` runs |

## What was run and verified

Verified on the tested environment: steps 2-9 (step 1 was done from a local copy rather than a fresh clone),
`scripts/ci.sh`, the pre-push hook installation,
`mlflow run . -e test --env-manager local`, drift-triggered retraining and `scripts/rollback.sh`.
Not run: `dvc exp run`, `mlflow run . --env-manager virtualenv`, the other MLflow Project entry points, the
optional `pipeline` container profile and running without Docker.
