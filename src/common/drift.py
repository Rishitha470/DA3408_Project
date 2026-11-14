"""Data drift detection with the Population Stability Index (PSI).

PSI = sum((actual% - expected%) * ln(actual% / expected%)) over bins.
Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, >= 0.2 significant shift.
"""
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.schema import FEATURES

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.2
EPS = 1e-4

# Fixed bin edges for continuous features (clinically meaningful for BMI)
NUMERIC_EDGES = {
    "BMI": [0, 18.5, 25, 30, 35, 40, 1000],
    "MentHlth": [-0.5, 0.5, 5, 10, 15, 20, 30.5],
    "PhysHlth": [-0.5, 0.5, 5, 10, 15, 20, 30.5],
}


def _bins(name: str, lo: float, hi: float) -> List[float]:
    """Bin edges: fixed for continuous features, one bin per integer level otherwise."""
    if name in NUMERIC_EDGES:
        return NUMERIC_EDGES[name]
    return [v - 0.5 for v in range(int(lo), int(hi) + 2)]


def _distribution(values: pd.Series, edges: List[float]) -> List[float]:
    counts, _ = np.histogram(values.astype(float), bins=edges)
    total = counts.sum()
    return (counts / total).tolist() if total else [0.0] * len(counts)


def build_reference(train_df: pd.DataFrame) -> Dict:
    """Per-feature bin edges and expected proportions computed on the training data."""
    ref = {}
    for f in FEATURES:
        edges = _bins(f.name, f.min, f.max)
        ref[f.name] = {"edges": edges, "expected": _distribution(train_df[f.name], edges)}
    return ref


def psi(expected: List[float], actual: List[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_drift(reference: Dict, current: pd.DataFrame, min_samples: int = 100) -> Dict:
    """Compare recent inputs with the training reference. Returns a JSON-serialisable report."""
    report = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "n_samples": int(len(current)), "min_samples": min_samples, "features": {}}
    if len(current) < min_samples:
        report.update(status="insufficient_data", drift_detected=False, drifted_features=[])
        return report
    drifted = []
    for name, ref in reference.items():
        if name not in current:
            continue
        value = psi(ref["expected"], _distribution(current[name], ref["edges"]))
        level = ("significant" if value >= PSI_SIGNIFICANT
                 else "moderate" if value >= PSI_MODERATE else "stable")
        report["features"][name] = {"psi": round(value, 4), "level": level}
        if level == "significant":
            drifted.append(name)
    report.update(status="ok", drift_detected=bool(drifted), drifted_features=drifted,
                  max_psi=max((v["psi"] for v in report["features"].values()), default=0.0))
    return report
