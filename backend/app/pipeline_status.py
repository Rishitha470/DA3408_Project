"""Builds the pipeline-console view: DVC DAG, stage status, run history, data and model reports."""
import json
from pathlib import Path
from typing import Dict, List

import yaml

DEFAULT_STAGES = ["ingest", "process", "train", "evaluate"]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _out_paths(stage: Dict) -> List[str]:
    paths = []
    for key in ("outs", "metrics", "plots"):
        for item in stage.get(key, []) or []:
            paths.append(next(iter(item)) if isinstance(item, dict) else item)
    return paths


def load_dag(dvc_file: Path) -> Dict:
    """Stages and edges of the DVC pipeline (edge A->B when B depends on an output of A)."""
    try:
        stages = yaml.safe_load(dvc_file.read_text())["stages"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return {"stages": [{"name": s, "cmd": None, "deps": []} for s in DEFAULT_STAGES],
                "edges": [[a, b] for a, b in zip(DEFAULT_STAGES, DEFAULT_STAGES[1:])]}
    outs = {name: _out_paths(st) for name, st in stages.items()}
    edges = []
    for name, st in stages.items():
        for dep in st.get("deps", []) or []:
            for src_name, src_outs in outs.items():
                if src_name != name and any(dep == o or dep.startswith(o.rstrip("/") + "/") for o in src_outs):
                    if [src_name, name] not in edges:
                        edges.append([src_name, name])
    return {"stages": [{"name": n, "cmd": st.get("cmd"), "deps": st.get("deps", [])}
                       for n, st in stages.items()], "edges": edges}


def build_status(runs: List[Dict], reports_dir: Path, dvc_file: Path, drift: Dict, tools: Dict) -> Dict:
    dag = load_dag(dvc_file)
    latest = {}
    for r in runs:                                   # runs are newest first
        latest.setdefault(r.get("stage"), r)
    for st in dag["stages"]:
        st["last_run"] = latest.get(st["name"])
    extra = [s for s in latest if s not in {st["name"] for st in dag["stages"]}]
    return {
        "dag": dag,
        "other_jobs": {s: latest[s] for s in extra},            # e.g. drift_check
        "runs": runs[:50],
        "summary": {"total": len(runs),
                    "success": sum(r.get("status") == "success" for r in runs),
                    "failed": sum(r.get("status") == "failed" for r in runs)},
        "data_quality": _read_json(reports_dir / "data_quality.json"),
        "model_comparison": _read_json(reports_dir / "model_comparison.json"),
        "test_metrics": _read_json(reports_dir / "metrics.json"),
        "acceptance": _read_json(reports_dir / "acceptance.json"),
        "drift": drift,
        "tools": tools,
    }
