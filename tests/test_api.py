"""API tests (FastAPI TestClient, local inference with a tiny model)."""
import pytest
import requests
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(model_env):
    from backend.app.main import app            # import after env vars are set by model_env
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["inference_mode"] == "local"


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_predict_valid_request(client, patient, model_env):
    r = client.post("/api/v1/predict", json=patient)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["probability"] <= 1
    assert b["prediction"] == int(b["probability"] >= b["threshold"])
    assert b["risk_level"] in {"low", "moderate", "high"}
    assert b["model_version"] == "test_lr@0000"
    assert "not a medical diagnosis" in b["disclaimer"]
    assert len(b["top_factors"]) <= 3


def test_healthy_person_scores_lower_than_high_risk_person(client, patient):
    healthy = dict(patient, HighBP=0, HighChol=0, BMI=22, Smoker=0, PhysActivity=1, Fruits=1,
                   GenHlth=1, MentHlth=0, PhysHlth=0, Age=3, Income=8)
    p_high = client.post("/api/v1/predict", json=patient).json()["probability"]
    p_low = client.post("/api/v1/predict", json=healthy).json()["probability"]
    assert p_low < p_high


@pytest.mark.parametrize("field, value", [("BMI", 5), ("Age", 14), ("HighBP", 2), ("GenHlth", 0),
                                          ("MentHlth", 31), ("Sex", "male"), ("Income", 2.5)])
def test_invalid_values_rejected(client, patient, field, value):
    r = client.post("/api/v1/predict", json=dict(patient, **{field: value}))
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert any(d["field"] == field for d in body["details"])


def test_missing_field_rejected(client, patient):
    patient.pop("Smoker")
    r = client.post("/api/v1/predict", json=patient)
    assert r.status_code == 422 and r.json()["details"][0]["field"] == "Smoker"


def test_unknown_field_rejected(client, patient):
    assert client.post("/api/v1/predict", json=dict(patient, Weight=80)).status_code == 422


def test_malformed_json_rejected(client):
    r = client.post("/api/v1/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_request_id_echoed_and_generated(client, patient):
    r = client.post("/api/v1/predict", json=patient, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123" and r.json()["request_id"] == "abc123"
    r2 = client.post("/api/v1/predict", json={})
    assert r2.headers["X-Request-ID"] == r2.json()["request_id"]


def test_model_unavailable_returns_503(client, patient, monkeypatch):
    from backend.app.main import state
    from backend.app.predictor import PredictorError

    def boom(rows):
        raise PredictorError("server down")
    monkeypatch.setattr(state.predictor, "predict_proba", boom)
    r = client.post("/api/v1/predict", json=patient)
    assert r.status_code == 503 and r.json()["error"] == "model_unavailable"


def test_unexpected_error_returns_generic_500(client, patient, monkeypatch):
    from backend.app.main import state

    def boom(rows):
        raise RuntimeError("SECRET internal detail")
    monkeypatch.setattr(state.predictor, "predict_proba", boom)
    r = client.post("/api/v1/predict", json=patient)
    assert r.status_code == 500 and "SECRET" not in r.text


def test_metrics_endpoint_exposes_monitoring(client, patient):
    client.post("/api/v1/predict", json=patient)
    text = client.get("/metrics").text
    for name in ("http_requests_total", "http_request_duration_seconds_bucket",
                 "model_predictions_total", "model_inference_duration_seconds",
                 "model_info", "data_drift_detected"):
        assert name in text
    assert 'endpoint="/metrics"' not in text


def test_predictions_are_logged_encrypted(client, patient, model_env):
    client.post("/api/v1/predict", json=patient)
    data = (model_env["log_dir"] / "predictions.enc").read_bytes()
    assert data and b"HighBP" not in data


def test_drift_endpoint(client, patient):
    for _ in range(25):
        client.post("/api/v1/predict", json=patient)
    report = client.get("/api/v1/drift").json()
    assert report["status"] == "ok" and "BMI" in report["features"]


def test_pipeline_endpoint_structure(client):
    b = client.get("/api/v1/pipeline").json()
    assert {"dag", "runs", "summary", "data_quality", "test_metrics", "drift", "tools"} <= set(b)
    assert len(b["dag"]["stages"]) == 4


def test_model_info(client):
    b = client.get("/api/v1/model").json()
    assert b["model_name"] == "test_lr" and b["test_metrics"]["roc_auc"] == 0.8


def test_reload_requires_admin_token(client):
    assert client.post("/api/v1/model/reload").status_code == 401
    r = client.post("/api/v1/model/reload", headers={"X-Admin-Token": "secret-token"})
    assert r.status_code == 200 and r.json()["status"] == "reloaded"


# ---------------------------------------------------------------- MLflow REST predictor
def test_mlflow_predictor_parses_response(model_env, monkeypatch, patient):
    import pandas as pd
    from backend.app.predictor import MlflowServerPredictor

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"predictions": [{"probability": 0.42, "risk_level": "moderate"}]}

    p = MlflowServerPredictor(model_env["model_dir"], "http://x/invocations")
    p.load()
    monkeypatch.setattr(p.session, "post", lambda *a, **k: Resp())
    assert p.predict_proba(pd.DataFrame([patient])).tolist() == [0.42]


def test_mlflow_predictor_connection_error(model_env, monkeypatch, patient):
    import pandas as pd
    from backend.app.predictor import MlflowServerPredictor, PredictorError

    def fail(*a, **k):
        raise requests.ConnectionError("refused")
    p = MlflowServerPredictor(model_env["model_dir"], "http://x/invocations")
    monkeypatch.setattr(p.session, "post", fail)
    with pytest.raises(PredictorError):
        p.predict_proba(pd.DataFrame([patient]))


def test_invalid_inference_mode_rejected(model_env):
    from backend.app.config import Settings
    from backend.app.predictor import make_predictor
    s = Settings()
    s.inference_mode = "magic"
    with pytest.raises(ValueError):
        make_predictor(s)
