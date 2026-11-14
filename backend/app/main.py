"""Diabetes Risk API - FastAPI backend.

Run locally:   uvicorn backend.app.main:app --port 8000          (INFERENCE_MODE=local by default)
Interactive API docs: http://localhost:8000/docs
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.app import metrics as m
from backend.app.config import settings
from backend.app.explain import top_factors, what_if_rows
from backend.app.pipeline_status import build_status
from backend.app.prediction_store import PredictionStore
from backend.app.predictor import BasePredictor, PredictorError, make_predictor
from backend.app.schemas import ErrorResponse, HealthResponse, PatientInput, PredictionResponse
from src.common.logging_utils import get_logger
from src.common.pipeline_log import read_runs
from src.common.schema import risk_level

log = get_logger("backend", str(settings.log_dir))


class AppState:
    """Mutable runtime state shared by the endpoints and the background refresher."""

    predictor: BasePredictor = None
    model_loaded: bool = False
    load_error: str = ""
    dependencies: dict = {}
    store: PredictionStore = None


state = AppState()


def load_model() -> None:
    """(Re)load the predictor; on failure the API stays up and reports 503 for predictions."""
    try:
        predictor = make_predictor(settings)
        predictor.load()
        state.predictor, state.model_loaded, state.load_error = predictor, True, ""
        m.update_model_gauges(predictor.meta, predictor.mode, settings.reports_dir)
        state.store.load_reference(settings.model_dir)
        log.info("model ready: %s (mode=%s, threshold=%.3f)", predictor.version, predictor.mode,
                 predictor.threshold)
    except (PredictorError, ValueError, OSError) as exc:
        state.model_loaded, state.load_error = False, str(exc)
        log.error("model not loaded: %s", exc)


def check_dependencies() -> dict:
    """Ping the other services (model server, MLflow, frontend, ...) for the dependency_up metric."""
    result = {}
    for name, url in settings.dependencies.items():
        try:
            ok = requests.get(url, timeout=2).status_code < 500
        except requests.RequestException:
            ok = False
        m.DEPENDENCY_UP.labels(name).set(1 if ok else 0)
        result[name] = ok
    return result


def refresh_monitoring() -> None:
    """Periodic job: pipeline gauges, live drift, dependency health, model metrics."""
    runs = read_runs(settings.reports_dir / "pipeline_runs.jsonl", limit=1000)
    m.update_pipeline_gauges(runs, settings.reports_dir)
    m.update_drift_gauges(state.store.drift_report())
    state.dependencies = check_dependencies()
    if state.model_loaded:
        m.update_model_gauges(state.predictor.meta, state.predictor.mode, settings.reports_dir)
    else:
        load_model()                                   # retry until the model becomes available


async def refresher() -> None:
    while True:
        try:
            await asyncio.to_thread(refresh_monitoring)
        except Exception as exc:                       # never let the loop die
            log.exception("monitoring refresh failed: %s", exc)
        await asyncio.sleep(settings.refresh_interval_s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.store = PredictionStore(settings.log_dir, settings.drift_window, settings.drift_min_samples)
    state.store.warm_start()
    load_model()
    task = asyncio.create_task(refresher())
    yield
    task.cancel()


app = FastAPI(title="Diabetes Risk Prediction API", version="1.0.0", lifespan=lifespan,
              description="Screening-only diabetes risk scores from 21 CDC health indicators.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"], expose_headers=["X-Request-ID"])


# ---------------------------------------------------------------- middleware & error handling
@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Request id, latency/count metrics and an access log line for every request."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    m.HTTP_IN_PROGRESS.inc()
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        m.HTTP_IN_PROGRESS.dec()
    elapsed = time.perf_counter() - t0
    route = request.scope.get("route")
    endpoint = route.path if route is not None else "unmatched"     # bounded label values
    if endpoint != "/metrics":
        m.HTTP_REQUESTS.labels(request.method, endpoint, response.status_code).inc()
        m.HTTP_LATENCY.labels(request.method, endpoint).observe(elapsed)
        log.info("%s %s %s %.1fms rid=%s", request.method, endpoint, response.status_code,
                 elapsed * 1000, rid)
    response.headers["X-Request-ID"] = rid
    return response


def _error(request: Request, status: int, error: str, message: str, details=None) -> JSONResponse:
    rid = getattr(request.state, "request_id", uuid.uuid4().hex)
    body = ErrorResponse(error=error, message=message, request_id=rid, details=details)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True),
                        headers={"X-Request-ID": rid})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    m.APP_EXCEPTIONS.labels("validation").inc()
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "msg": e["msg"]}
               for e in exc.errors()]
    return _error(request, 422, "validation_failed", "Some answers are missing or out of range.", details)


@app.exception_handler(PredictorError)
async def predictor_error(request: Request, exc: PredictorError):
    m.APP_EXCEPTIONS.labels("model_unavailable").inc()
    log.error("prediction failed: %s", exc)
    return _error(request, 503, "model_unavailable",
                  "The prediction service is temporarily unavailable. Please try again shortly.")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    m.APP_EXCEPTIONS.labels(type(exc).__name__).inc()
    log.exception("unhandled error: %s", exc)
    return _error(request, 500, "internal_error", "Something went wrong on our side.")


# ---------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness: the API process is up (model may still be loading)."""
    return HealthResponse(status="ok", model_loaded=state.model_loaded,
                          model_version=state.predictor.version if state.model_loaded else None,
                          inference_mode=settings.inference_mode, dependencies=state.dependencies)


@app.get("/ready", tags=["ops"])
def ready(request: Request):
    """Readiness: 200 only when a model is loaded and predictions can be served."""
    if not state.model_loaded:
        return _error(request, 503, "not_ready", state.load_error or "model not loaded")
    return {"status": "ready", "model_version": state.predictor.version}


@app.get("/metrics", tags=["ops"])
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"],
          responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def predict(patient: PatientInput, request: Request):
    """Diabetes risk score, risk level and the top changeable risk factors for one person."""
    if not state.model_loaded:
        raise PredictorError(state.load_error or "model not loaded")
    predictor = state.predictor
    inputs = {k: float(v) for k, v in patient.model_dump().items()}
    rows, changed = what_if_rows(inputs)
    t0 = time.perf_counter()
    scores = predictor.predict_proba(rows)           # original + what-if rows in one call
    m.INFERENCE_LATENCY.observe(time.perf_counter() - t0)

    probability = float(scores[0])
    level = risk_level(probability, predictor.threshold)
    m.PREDICTIONS.labels(level).inc()
    m.PREDICTION_SCORE.observe(probability)
    result = PredictionResponse(
        request_id=request.state.request_id, probability=round(probability, 4),
        prediction=int(probability >= predictor.threshold), risk_level=level,
        threshold=round(predictor.threshold, 4), top_factors=top_factors(scores, changed),
        model_name=predictor.meta.get("model_name", "unknown"), model_version=predictor.version)
    state.store.record(inputs, {"probability": result.probability, "risk_level": level})
    return result


@app.get("/api/v1/model", tags=["model"])
def model_info(request: Request):
    """Model card: name, version, threshold, validation/test metrics, feature importance."""
    if not state.model_loaded:
        return _error(request, 503, "model_unavailable", state.load_error or "model not loaded")
    meta = dict(state.predictor.meta)
    meta["inference_mode"] = state.predictor.mode
    meta["test_metrics"] = m._read_json(settings.reports_dir / "metrics.json") or None
    return meta


@app.post("/api/v1/model/reload", tags=["model"])
def reload_model(request: Request, x_admin_token: str = Header(default="")):
    """Reload the model after retraining (requires X-Admin-Token when ADMIN_TOKEN is set)."""
    if settings.admin_token and x_admin_token != settings.admin_token:
        return _error(request, 401, "unauthorized", "invalid admin token")
    load_model()
    if not state.model_loaded:
        return _error(request, 503, "model_unavailable", state.load_error)
    return {"status": "reloaded", "model_version": state.predictor.version}


@app.get("/api/v1/pipeline", tags=["pipeline"])
def pipeline():
    """Pipeline console data: DVC DAG, last run per stage, run history, data & model reports."""
    runs = read_runs(settings.reports_dir / "pipeline_runs.jsonl", limit=200)
    return build_status(runs, settings.reports_dir, settings.dvc_file,
                        state.store.last_report, settings.tool_links)


@app.get("/api/v1/drift", tags=["monitoring"])
def drift():
    """Latest data-drift report (PSI per feature) computed from recent prediction inputs."""
    return state.store.drift_report()
