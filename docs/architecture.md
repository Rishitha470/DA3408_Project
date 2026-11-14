# Architecture

## Diagram

```mermaid
flowchart LR
  subgraph Offline["Offline: training pipeline (DVC DAG, run on host or in the pipeline container)"]
    UCI[(UCI / CDC dataset<br/>253,680 rows)] --> ING[ingest]
    ING --> PRO["process<br/>(Ray Data: validate, clean, split)"]
    PRO --> TRN["train<br/>(LR, RF, XGBoost)"]
    TRN --> EVA["evaluate<br/>(test set + acceptance gate)"]
    PRO --> EVA
  end

  subgraph Tracking["Experiment tracking"]
    MLF[("MLflow tracking server<br/>+ Model Registry :5000")]
  end

  subgraph Online["Online: Docker Compose services"]
    FE["frontend<br/>nginx static UI :8080"] -- "REST (configurable API_BASE_URL)" --> BE["backend<br/>FastAPI :8000"]
    BE -- "REST /invocations" --> MS["model-server<br/>mlflow models serve :5001"]
    PROM["Prometheus :9090"] -- scrape /metrics --> BE
    PROM -- scrape --> NE["node-exporter :9100"]
    PROM -- alerts --> AM["Alertmanager :9093"]
    GRA["Grafana :3000"] -- PromQL --> PROM
  end

  TRN -- "params, metrics, plots, model" --> MLF
  EVA -- "register @champion" --> MLF
  TRN -- "models/ (DVC output)" --> MS
  EVA -- "reports/*.json" --> BE
  BE -- "encrypted inputs<br/>logs/predictions.enc" --> DM["drift monitor<br/>(PSI)"]
  DM -- "drift -> dvc repro" --> ING
  USER((User)) --> FE
```

## Blocks

| Block | Technology | Responsibility |
|---|---|---|
| ingest | Python, ucimlrepo | Downloads the CDC Diabetes Health Indicators data (UCI id 891), checks the schema, writes `data/raw/diabetes.csv` and its MD5. |
| process | **Ray Data** | Splits the CSV into blocks processed in parallel: data-quality checks (nulls, ranges, non-integer codes, labels), quality gate, cleaning, stratified 70/15/15 split, drift reference distribution. Reports throughput. |
| train | scikit-learn, XGBoost, **MLflow** | Trains 6 candidates with class-imbalance handling, tunes the decision threshold for recall (F2), logs every run (params, metrics, plots, Git commit, data hash), selects the best by validation ROC-AUC, saves the model bundle and an MLflow pyfunc model. |
| evaluate | MLflow Model Registry | Scores the held-out test set, writes DVC metrics/plots, applies the acceptance gate, and registers the model as `diabetes-risk-model@champion` only if it passes. |
| DVC | **DVC** | Defines the DAG (`dvc.yaml`), caches data/models, re-runs only changed stages, versions metrics and plots. `scripts/ci.sh` is the CI gate. |
| MLflow tracking server | MLflow | Stores runs and artifacts (proxied), hosts the Model Registry and the UI. |
| model-server | `mlflow models serve` | The inference engine: serves `models/mlflow_model` over REST. |
| backend | **FastAPI** | Validates input, calls the model server, explains the result (top changeable factors), logs inputs encrypted, computes live drift, exposes Prometheus metrics, serves pipeline status to the UI. |
| frontend | nginx, HTML/CSS/JS | Step-by-step risk questionnaire, result page, pipeline console, user manual. Knows the backend only through a configurable URL. |
| Prometheus / Alertmanager | Prometheus | Scrapes metrics every 5 s, evaluates 7 alert rules (backend down, model server down, error rate, latency, drift, pipeline failure, risk-share spike). |
| Grafana | Grafana | Provisioned dashboard (26 panels): service health, traffic, latency, errors, model outputs, drift per feature, pipeline stages, CPU/memory. |
| drift monitor | Python (PSI) | Batch job that decrypts recent inputs, compares them with the training distribution and triggers `dvc repro` on drift. |

## Data flow of one prediction

1. The browser sends the 21 answers as JSON to `POST /api/v1/predict` on the backend.
2. The backend validates them (Pydantic). Invalid input returns 422 with the field names.
3. It builds the original row plus one "what-if" row per changeable risk factor and sends all rows in **one** request to the model server.
4. The model server runs the scikit-learn pipeline (feature engineering + classifier) and returns a risk score per row.
5. The backend derives the risk level from the tuned threshold, ranks the factors, records metrics, appends the encrypted input to the prediction log and returns the result.
