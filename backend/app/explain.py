"""Per-person explanation: which changeable factors raise this person's risk the most?

Model-agnostic "what-if" analysis: for every modifiable factor that is not at its healthier value,
re-score the person with only that factor changed. The drop in risk score is the factor's impact.
All variants are scored in ONE batch call, so it costs a single model-server request.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.schema import FEATURES

MIN_IMPACT = 0.005      # ignore changes smaller than half a percentage point


def what_if_rows(row: Dict[str, float]) -> Tuple[pd.DataFrame, List]:
    """The original row followed by one modified row per changeable factor."""
    rows, changed = [dict(row)], []
    for f in FEATURES:
        if not f.modifiable or f.healthy is None:
            continue
        current = float(row[f.name])
        healthier = float(f.healthy)
        # BMI: only "improve" when above the healthy reference; never push a low BMI upwards
        if f.name == "BMI" and current <= healthier:
            continue
        if current == healthier:
            continue
        variant = dict(row)
        variant[f.name] = healthier
        rows.append(variant)
        changed.append((f, current, healthier))
    return pd.DataFrame(rows), changed


def top_factors(scores: np.ndarray, changed: List, k: int = 3) -> List[Dict]:
    """Rank factors by how much the risk score drops when they are improved."""
    base = float(scores[0])
    factors = []
    for (f, current, healthier), score in zip(changed, scores[1:]):
        impact = base - float(score)
        if impact >= MIN_IMPACT:
            factors.append({"feature": f.name, "label": f.label, "current_value": current,
                            "healthier_value": healthier, "impact": round(impact, 4)})
    return sorted(factors, key=lambda x: x["impact"], reverse=True)[:k]
