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
| Drift | PSI per feature, live in the backend + batch job | Simple, well-understood thresholds (0.1 / 0.2); visible in Grafana in near real time. | KS test: less interpretable for categorical codes. |
| Security | Encrypted input log (Fernet), no personal identifiers, non-root containers, CORS allow-list, admin token for reload | Health data is sensitive. | |
| Packaging | Docker Compose: frontend, backend, model-server, MLflow, Prometheus, Alertmanager, Grafana, node-exporter | One command starts everything; identical environments. | Kubernetes: unnecessary for one machine. |

## 4. Software paradigm

The code is **object-oriented where there is state or substitutable behaviour** and functional elsewhere:

* `BasePredictor` (abstract) with `LocalPredictor` and `MlflowServerPredictor` (Strategy pattern, chosen by
  `make_predictor` factory from `INFERENCE_MODE`).
* `PredictionStore` encapsulates the encrypted log, the recent-input window and drift computation.
* `Settings` (dataclass) holds all configuration from environment variables.
* Pipeline stages, metrics and drift maths are pure functions, which makes them easy to unit test.

## 5. Lifecycle (AIOps)

1. **Develop:** change code or `params.yaml` -> `scripts/ci.sh` (lint, 73 tests, `dvc repro`, metrics diff).
2. **Train:** DVC runs only the affected stages; every run is in MLflow with the Git commit.
3. **Gate:** the evaluate stage registers the model as `@champion` only if acceptance criteria hold.
4. **Deploy:** `docker compose up -d --build`; `models/` is mounted into the model server.
5. **Monitor:** Prometheus + Grafana + alert rules; live PSI drift per feature.
6. **Retrain:** `python -m src.monitoring.drift_monitor --retrain` runs `dvc repro --force` when drift is found.
7. **Roll back:** `scripts/rollback.sh <commit>` restores the earlier `models/` from the DVC cache and restarts the services.

## 6. Reproducibility

Every MLflow run carries `git_commit`, `git_dirty` and `train_data_md5` tags. `dvc.lock` records the hash of
every input and output. To reproduce a result: `git checkout <commit> && dvc checkout` (or `dvc repro`).
Seeds are fixed in `params.yaml`.

## 7. Limitations

* The score ranks risk; it is not a calibrated probability (class weighting shifts scores upwards). The UI says so.
* The model learned from US survey data from 2015 and may transfer imperfectly to other populations.
* Retraining on drift re-ingests the public dataset; in a real deployment newly labelled data would be added.
