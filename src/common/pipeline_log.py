"""Records every pipeline stage execution (success or failure) to reports/pipeline_runs.jsonl.

The backend reads this file to power the pipeline console in the UI and the Prometheus
pipeline metrics (duration, throughput, status per stage).
"""
import json
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List

from src.common.config import REPORTS_DIR

RUNS_FILE = REPORTS_DIR / "pipeline_runs.jsonl"


class StageRecord(dict):
    """Mutable record a stage can enrich (rows processed, extra info) while it runs."""

    def set_rows(self, rows_in: int, rows_out: int = None) -> None:
        self["rows_in"] = int(rows_in)
        self["rows_out"] = int(rows_out if rows_out is not None else rows_in)


@contextmanager
def track_stage(stage: str, runs_file: Path = RUNS_FILE) -> Iterator[StageRecord]:
    """Context manager: times a stage and appends a JSON line with its outcome."""
    rec = StageRecord(run_id=uuid.uuid4().hex[:12], stage=stage, status="running",
                      started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    t0 = time.perf_counter()
    try:
        yield rec
        rec["status"] = "success"
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
        raise
    finally:
        rec["duration_s"] = round(time.perf_counter() - t0, 3)
        rec["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if rec.get("rows_out") and rec["duration_s"] > 0:
            rec["rows_per_s"] = round(rec["rows_out"] / rec["duration_s"], 1)
        runs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(runs_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(rec)) + "\n")


def read_runs(runs_file: Path = RUNS_FILE, limit: int = 100) -> List[Dict]:
    """Return the most recent stage records (newest first). Corrupt lines are skipped."""
    if not runs_file.exists():
        return []
    out = []
    for line in runs_file.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))[:limit]
