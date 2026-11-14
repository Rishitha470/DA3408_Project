"""Send realistic (or deliberately shifted) traffic to the API - for demos of monitoring and drift.

    python scripts/simulate_traffic.py --n 300                 # replays real held-out test rows -> no drift
    python scripts/simulate_traffic.py --n 300 --drift         # same rows, older and heavier -> drift alert
"""
import argparse
import random
import time
from pathlib import Path

import requests

TEST_SPLIT = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.parquet"
FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]


def load_rows():
    """Real survey rows from the held-out test split (requires `dvc repro` to have run)."""
    import pandas as pd
    df = pd.read_parquet(TEST_SPLIT)[FEATURES]
    return [{k: (float(v) if k == "BMI" else int(v)) for k, v in r.items()} for r in df.to_dict("records")]


def shift(row: dict) -> dict:
    """Simulated population change: older, heavier, less active people."""
    row = dict(row)
    row["BMI"] = min(98.0, row["BMI"] + 9)
    row["Age"] = min(13, row["Age"] + 3)
    row["PhysActivity"] = 0
    return row


def random_patient(rng, drift: bool = False) -> dict:
    """A plausible survey answer; with drift=True the population is older with higher BMI."""
    age = rng.randint(9, 13) if drift else rng.choice([2, 4, 5, 6, 7, 8, 9, 10, 11])
    bmi = round(rng.uniform(33, 48) if drift else min(max(rng.gauss(28, 6), 15), 60), 1)
    yes = (lambda p: int(rng.random() < p))
    return {"HighBP": yes(0.7 if drift else 0.43), "HighChol": yes(0.6 if drift else 0.42),
            "CholCheck": yes(0.96), "BMI": bmi, "Smoker": yes(0.44), "Stroke": yes(0.04),
            "HeartDiseaseorAttack": yes(0.09), "PhysActivity": yes(0.4 if drift else 0.76),
            "Fruits": yes(0.63), "Veggies": yes(0.81), "HvyAlcoholConsump": yes(0.06),
            "AnyHealthcare": yes(0.95), "NoDocbcCost": yes(0.08),
            "GenHlth": rng.randint(3, 5) if drift else rng.randint(1, 4),
            "MentHlth": rng.choice([0, 0, 0, 2, 5, 10, 30]), "PhysHlth": rng.choice([0, 0, 0, 3, 7, 15, 30]),
            "DiffWalk": yes(0.17), "Sex": yes(0.44), "Age": age,
            "Education": rng.randint(3, 6), "Income": rng.randint(1, 8)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--delay", type=float, default=0.02, help="seconds between requests")
    a = ap.parse_args()
    rng, s, codes = random.Random(), requests.Session(), {}
    rows = load_rows() if TEST_SPLIT.exists() else None
    if rows is None:
        print("data/processed/test.parquet not found - using a generated population instead")
    for i in range(a.n):
        if rows:
            body = rng.choice(rows)
            body = shift(body) if a.drift else body
        else:
            body = random_patient(rng, a.drift)
        try:
            code = s.post(f"{a.url}/api/v1/predict", json=body, timeout=10).status_code
        except requests.RequestException:
            code = "conn_error"
        codes[code] = codes.get(code, 0) + 1
        time.sleep(a.delay)
    print(f"sent {a.n} requests (drift={a.drift}): {codes}")
