"""Generates monitoring/grafana/dashboards/diabetes-risk.json (run again after editing panels)."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels, _id = [], [0]


def nid():
    _id[0] += 1
    return _id[0]


def row(title, y):
    panels.append({"type": "row", "title": title, "id": nid(), "collapsed": False,
                   "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})


def target(expr, legend=""):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": chr(65 + len(legend) % 26)}


def panel(kind, title, x, y, w, h, targets, unit=None, desc="", **extra):
    defaults = {"unit": unit} if unit else {}
    defaults.update(extra.pop("defaults", {}))
    p = {"type": kind, "title": title, "description": desc, "id": nid(), "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [dict(t, refId=chr(65 + i)) for i, t in enumerate(targets)],
         "fieldConfig": {"defaults": defaults, "overrides": []}, "options": extra.pop("options", {})}
    p.update(extra)
    panels.append(p)


UPDOWN = {"mappings": [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"},
                                                     "1": {"text": "UP", "color": "green"}}}],
          "thresholds": {"mode": "absolute",
                         "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}}
PSI = {"min": 0, "thresholds": {"mode": "absolute", "steps": [
    {"color": "green", "value": None}, {"color": "orange", "value": 0.1}, {"color": "red", "value": 0.2}]}}

row("Service health", 0)
panel("stat", "Backend scrape", 0, 1, 4, 4, [target('up{job="backend"}')], defaults=UPDOWN,
      desc="Prometheus can scrape the FastAPI backend")
panel("stat", "Dependencies", 4, 1, 14, 4, [target("dependency_up", "{{service}}")], defaults=UPDOWN,
      desc="Health checks run by the backend against every other service")
panel("stat", "Served model", 18, 1, 6, 4, [target("model_info", "{{model_version}} ({{inference_mode}})")],
      options={"textMode": "name", "colorMode": "none"})

row("API traffic", 5)
panel("timeseries", "Requests per second by endpoint", 0, 6, 8, 8,
      [target("sum by (endpoint, status) (rate(http_requests_total[1m]))", "{{endpoint}} {{status}}")], "reqps")
LAT = 'sum by (le) (rate(http_request_duration_seconds_bucket{endpoint="/api/v1/predict"}[1m]))'
panel("timeseries", "Prediction latency (p50 / p95 / p99)", 8, 6, 8, 8, [
    target(f"histogram_quantile(0.50, {LAT})", "p50"),
    target(f"histogram_quantile(0.95, {LAT})", "p95"),
    target(f"histogram_quantile(0.99, {LAT})", "p99")], "s")
panel("timeseries", "Errors (5xx share) and handled exceptions", 16, 6, 8, 8, [
    target("job:http_error_ratio:rate5m", "5xx share of predictions"),
    target("sum by (type) (rate(app_exceptions_total[1m]))", "exceptions/s {{type}}")], "percentunit")

row("Model", 14)
panel("timeseries", "Predictions by risk level", 0, 15, 8, 8,
      [target("sum by (risk_level) (rate(model_predictions_total[1m]))", "{{risk_level}}")], "reqps",
      fieldConfig_custom=None)
panel("bargauge", "Risk score distribution (last 1h)", 8, 15, 8, 8,
      [target("sum by (le) (increase(model_prediction_probability_bucket[1h]))", "<= {{le}}")],
      options={"orientation": "vertical", "displayMode": "gradient"},
      desc="Cumulative histogram buckets of the model's risk score")
panel("stat", "Test-set metrics of served model", 16, 15, 8, 4, [target("model_test_metric", "{{metric}}")],
      defaults={"decimals": 3})
panel("timeseries", "Inference latency p95", 16, 19, 8, 4,
      [target("histogram_quantile(0.95, sum by (le) (rate(model_inference_duration_seconds_bucket[1m])))", "p95")], "s")

row("Data drift (PSI vs. training data)", 23)
panel("stat", "Drift detected", 0, 24, 4, 4, [target("data_drift_detected")], defaults={
    "mappings": [{"type": "value", "options": {"0": {"text": "No", "color": "green"},
                                               "1": {"text": "YES", "color": "red"}}}],
    "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}})
panel("stat", "Inputs in drift window", 0, 28, 4, 4, [target("data_drift_window_samples")])
panel("bargauge", "PSI per feature (0.1 moderate, 0.2 significant)", 4, 24, 20, 8,
      [target("sort_desc(data_drift_psi)", "{{feature}}")], defaults=PSI,
      options={"orientation": "horizontal", "displayMode": "gradient"})

row("Training & data pipeline", 32)
panel("stat", "Last run status per stage", 0, 33, 8, 4, [target("pipeline_stage_last_status", "{{stage}}")],
      defaults={"mappings": [{"type": "value", "options": {"0": {"text": "FAILED", "color": "red"},
                                                           "1": {"text": "OK", "color": "green"}}}]})
panel("bargauge", "Stage duration (s)", 8, 33, 8, 8, [target("pipeline_stage_last_duration_seconds", "{{stage}}")], "s",
      options={"orientation": "horizontal"})
panel("bargauge", "Stage throughput (rows/s)", 16, 33, 8, 8, [target("pipeline_stage_rows_per_second", "{{stage}}")],
      options={"orientation": "horizontal"})
panel("stat", "Ray processing throughput (rows/s)", 0, 37, 4, 4, [target("data_pipeline_ray_rows_per_second")])
panel("stat", "Rows in / kept", 4, 37, 4, 4, [target("data_pipeline_rows", "{{kind}}")])

row("Infrastructure (node-exporter)", 41)
panel("timeseries", "CPU usage", 0, 42, 12, 7,
      [target('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))', "CPU busy")], "percentunit")
panel("timeseries", "Memory usage", 12, 42, 12, 7,
      [target("1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes", "memory used")], "percentunit")

for p in panels:                 # remove helper-only keys
    p.pop("fieldConfig_custom", None)

dashboard = {"uid": "diabetes-risk", "title": "Diabetes Risk - AIOps monitoring", "schemaVersion": 39,
             "version": 1, "editable": True, "refresh": "5s", "time": {"from": "now-30m", "to": "now"},
             "tags": ["aiops", "diabetes"], "panels": panels, "templating": {"list": []}, "annotations": {"list": []}}
out = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards" / "diabetes-risk.json"
out.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {out} ({len(panels)} panels)")
