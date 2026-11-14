"""Synthetic data with the exact schema of the CDC dataset.

Used ONLY for unit tests and for offline smoke runs (params: data.source = synthetic).
The real project uses the CDC Diabetes Health Indicators data (data.source = uci).
"""
import numpy as np
import pandas as pd

from src.common.schema import ALL_COLUMNS, TARGET


def make_synthetic(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.choice(np.arange(1, 14), size=n, p=np.array([2, 3, 4, 5, 6, 8, 10, 12, 13, 12, 10, 8, 7]) / 100)
    bmi = np.clip(np.round(rng.lognormal(np.log(27.5), 0.2, n)), 12, 98)
    high_bp = (rng.random(n) < 0.15 + 0.035 * age).astype(int)
    high_chol = (rng.random(n) < 0.2 + 0.025 * age).astype(int)
    phys = (rng.random(n) < 0.78).astype(int)
    gen = np.clip(np.round(rng.normal(2.5, 1.0, n) + 0.02 * (bmi - 27)), 1, 5)
    df = pd.DataFrame({
        "HighBP": high_bp, "HighChol": high_chol,
        "CholCheck": (rng.random(n) < 0.96).astype(int), "BMI": bmi,
        "Smoker": (rng.random(n) < 0.44).astype(int),
        "Stroke": (rng.random(n) < 0.04).astype(int),
        "HeartDiseaseorAttack": (rng.random(n) < 0.03 + 0.008 * age).astype(int),
        "PhysActivity": phys,
        "Fruits": (rng.random(n) < 0.63).astype(int), "Veggies": (rng.random(n) < 0.81).astype(int),
        "HvyAlcoholConsump": (rng.random(n) < 0.06).astype(int),
        "AnyHealthcare": (rng.random(n) < 0.95).astype(int),
        "NoDocbcCost": (rng.random(n) < 0.08).astype(int),
        "GenHlth": gen,
        "MentHlth": np.where(rng.random(n) < 0.7, 0, rng.integers(1, 31, n)),
        "PhysHlth": np.where(rng.random(n) < 0.65, 0, rng.integers(1, 31, n)),
        "DiffWalk": (rng.random(n) < 0.08 + 0.02 * (gen - 1)).astype(int),
        "Sex": (rng.random(n) < 0.44).astype(int), "Age": age,
        "Education": rng.choice(np.arange(1, 7), size=n, p=[.01, .03, .05, .25, .28, .38]),
        "Income": rng.choice(np.arange(1, 9), size=n, p=[.04, .05, .07, .08, .10, .14, .17, .35]),
    })
    logit = (-5.9 + 0.12 * (bmi - 27) + 0.95 * high_bp + 0.7 * high_chol + 0.7 * (gen - 1)
             + 0.21 * age - 0.3 * phys + 0.5 * df["HeartDiseaseorAttack"] + 0.4 * df["DiffWalk"]
             - 0.08 * df["Income"] + 0.25 * df["Sex"])
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df[ALL_COLUMNS].astype(float)
