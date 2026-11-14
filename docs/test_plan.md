# Test Plan

## 1. Scope and objectives

Verify that the data pipeline, the training/evaluation logic, the REST API and the user interface behave
as specified in the LLD, and that the deployed model meets the acceptance criteria.

In scope: data validation and cleaning (Ray), feature engineering, training utilities, acceptance gate,
drift detection, encryption, pipeline run logging, DVC DAG parsing, all API endpoints, error handling,
the MLflow-server client, and the user interface flow.
Out of scope: load testing at production scale; clinical validation of the model.

## 2. Test levels and tools

| Level | What | Tool | How to run |
|---|---|---|---|
| Unit | Pure functions and classes | pytest | `python scripts/run_tests.py` |
| Integration | Ray pipeline end to end; API with a real model (TestClient) | pytest, Ray, FastAPI TestClient | same |
| Pipeline | Full DVC DAG incl. acceptance gate | DVC | `dvc repro` |
| System | All containers running together | Docker Compose, curl, browser | Section 5 |
| Performance | API throughput and latency | `scripts/load_test.py` | Section 5 |
| UI / usability | Questionnaire flow, validation, result, pipeline console | Manual checklist | Section 6 |

The automated suite needs no network, Docker or MLflow server: it uses a small synthetic dataset with the
real schema and a tiny model trained inside the test fixture.

## 3. Acceptance criteria

| # | Criterion | Threshold | Verified by |
|---|---|---|---|
| AC1 | All automated tests pass | 100 % | `scripts/run_tests.py` -> `docs/test_report.md` |
| AC2 | Test-set ROC-AUC | >= 0.78 | evaluate stage (`reports/acceptance.json`) |
| AC3 | Test-set recall | >= 0.70 | evaluate stage |
| AC4 | Test-set precision | >= 0.25 | evaluate stage |
| AC5 | Invalid input never reaches the model and returns 422 with the field name | always | TC-API-05..09 |
| AC6 | Model outage returns 503 with a friendly message (no crash, no internals) | always | TC-API-10, 11; system test S4 |
| AC7 | p95 latency of `/api/v1/predict` at 4 concurrent users | < 500 ms | `scripts/load_test.py` |
| AC8 | A non-technical user completes the questionnaire without help | 5 of 5 testers | UI checklist |

The model is registered as `@champion` only when AC2-AC4 hold (automatic gate).

## 4. Automated test cases

| ID | File | Test case | Expected result |
|---|---|---|---|
| TC-CORE-01 | test_core | schema has 21 unique features | 21 names, no duplicates |
| TC-CORE-02 | test_core | healthy reference values within feature ranges | all inside |
| TC-CORE-03 | test_core | risk level bands follow the threshold (6 boundary cases) | low / moderate / high as specified |
| TC-CORE-04 | test_core | engineered columns appended | 25 columns |
| TC-CORE-05 | test_core | engineered values correct (BMI class, counts, cap 30) | exact values |
| TC-CORE-06 | test_core | feature engineering does not mutate input | input unchanged |
| TC-CORE-07 | test_core | encrypted log round trip | records restored; plaintext not on disk |
| TC-CORE-08 | test_core | wrong key cannot read log | no records |
| TC-CORE-09 | test_core | no key -> no cipher | None |
| TC-CORE-10 | test_core | stage run recorded on success | status success, rows, duration |
| TC-CORE-11 | test_core | stage failure recorded and re-raised | status failed, error text |
| TC-CORE-12 | test_core | run log newest first, corrupt lines skipped | correct order |
| TC-CORE-13 | test_core | DVC DAG edges parsed from dvc.yaml | ingest->process->train->evaluate |
| TC-CORE-14 | test_core | DAG fallback when dvc.yaml missing | default 4 stages |
| TC-DATA-01 | test_data_pipeline | invalid rows flagged (null, range, non-integer, label) | 4 flagged |
| TC-DATA-02 | test_data_pipeline | Arrow-backed columns accepted | no false flags |
| TC-DATA-03 | test_data_pipeline | non-numeric values invalid | flagged |
| TC-DATA-04 | test_data_pipeline | per-block quality counts | exact counts |
| TC-DATA-05 | test_data_pipeline | cleaning drops invalid rows, casts label to int | 2 rows kept |
| TC-DATA-06 | test_data_pipeline | quality gate (pass + 3 failure modes) | pass / DataQualityError |
| TC-DATA-07 | test_data_pipeline | synthetic ingest matches schema | 22 columns |
| TC-DATA-08 | test_data_pipeline | unknown data source rejected | ValueError |
| TC-DATA-09 | test_data_pipeline | Ray pipeline end to end with bad rows | 506 in, 4 invalid, 502 out |
| TC-DATA-10 | test_data_pipeline | seeded split identical when rows arrive in a different order (Ray) | same train/val/test rows |
| TC-MOD-01 | test_modeling | LR, RF, XGBoost each train and predict | scores in [0, 1] |
| TC-MOD-02 | test_modeling | unknown model type rejected | ValueError |
| TC-MOD-03 | test_modeling | higher beta gives recall >= lower beta | holds |
| TC-MOD-04 | test_modeling | metrics dictionary complete | 6 keys |
| TC-MOD-05 | test_modeling | feature importance normalised | sums to 1, 25 features |
| TC-MOD-06 | test_modeling | acceptance gate pass / fail | correct verdict per criterion |
| TC-MOD-07 | test_modeling | PSI of identical distributions | 0 |
| TC-MOD-08 | test_modeling | no drift on same distribution | drift_detected false |
| TC-MOD-09 | test_modeling | BMI shift detected | BMI in drifted features |
| TC-MOD-10 | test_modeling | too few samples | insufficient_data |
| TC-MOD-11 | test_modeling | what-if only on changeable, unhealthy factors | Age/Sex/Veggies excluded |
| TC-MOD-12 | test_modeling | low BMI never pushed up | BMI excluded |
| TC-MOD-13 | test_modeling | top factors sorted, filtered, limited to 3 | as specified |
| TC-MOD-14 | test_modeling | model registration retried after a tracking-server disconnect | succeeds on 2nd attempt |
| TC-API-01 | test_api | `/health` | 200, model loaded |
| TC-API-02 | test_api | `/ready` | 200 |
| TC-API-03 | test_api | valid prediction | 200, consistent prediction/threshold, disclaimer |
| TC-API-04 | test_api | healthy profile scores lower than high-risk profile | p_low < p_high |
| TC-API-05 | test_api | 7 invalid values (range, type, non-integer) | 422 naming the field |
| TC-API-06 | test_api | missing field | 422 naming the field |
| TC-API-07 | test_api | unknown field | 422 |
| TC-API-08 | test_api | malformed JSON | 422 |
| TC-API-09 | test_api | request id echoed / generated | header = body |
| TC-API-10 | test_api | model unavailable | 503 `model_unavailable` |
| TC-API-11 | test_api | unexpected error | 500, no internal details |
| TC-API-12 | test_api | `/metrics` exposes API, model, drift metrics; not self-counted | names present |
| TC-API-13 | test_api | prediction log encrypted | file exists, no plaintext |
| TC-API-14 | test_api | drift endpoint after traffic | status ok, per-feature PSI |
| TC-API-15 | test_api | pipeline endpoint structure | all keys, 4 stages |
| TC-API-16 | test_api | model info | name and test metrics |
| TC-API-17 | test_api | reload requires admin token | 401 then 200 |
| TC-API-18 | test_api | MLflow client parses server response | scores returned |
| TC-API-19 | test_api | MLflow client connection error | PredictorError |
| TC-API-20 | test_api | invalid INFERENCE_MODE | ValueError |

Parametrised cases expand this list to 74 executed tests (see `docs/test_report.md`).

## 5. System test checklist (Docker)

| ID | Step | Expected |
|---|---|---|
| S1 | `docker compose ps` | all services `running` / `healthy` |
| S2 | `curl localhost:8000/health` | `model_loaded: true`, `inference_mode: mlflow`, all dependencies `true` |
| S3 | Complete the questionnaire at `localhost:8080` | result with level, dial, factors, disclaimer |
| S4 | `docker compose stop model-server`, submit again | friendly "starting up or busy" message; Grafana shows model-server DOWN; alert `ModelServerDown` fires; restart fixes it |
| S5 | `python scripts/simulate_traffic.py --n 300` | Grafana traffic panels move; drift "No" |
| S6 | `python scripts/simulate_traffic.py --n 400 --drift` | drift "YES" for BMI/Age/PhysActivity; alert `DataDriftDetected` |
| S7 | `python scripts/load_test.py --concurrency 1 4 8` | table recorded in the report; AC7 checked |
| S8 | Pipeline console at `localhost:8080/pipeline.html` | DAG with statuses, run history, metrics, drift |

## 6. UI checklist (manual)

Pressing Next with unanswered questions highlights them; the BMI is computed from height/weight; out-of-range
numbers are refused; Back keeps answers; works at 390 px width (phone); keyboard-only use works (Tab, Enter, Space);
backend stopped -> clear error, no blank page.

## 7. Defect handling

Failed tests block a commit (`scripts/ci.sh`, optional pre-push hook). The test report lists every case with
its result; failed pipeline stages appear in the pipeline console and as the `PipelineStageFailed` alert.
