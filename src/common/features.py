"""Feature engineering shared by training and serving.

The function is placed inside the scikit-learn Pipeline (FunctionTransformer), so the exact same
code runs at training time and at inference time -> no training/serving skew.
"""
import numpy as np
import pandas as pd

from src.common.schema import FEATURE_NAMES

ENGINEERED = ["bmi_category", "cardio_risk_count", "lifestyle_score", "poor_health_days"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (raw features) with engineered features appended."""
    out = pd.DataFrame(df, columns=FEATURE_NAMES).astype(float).copy()
    # WHO BMI classes: 0 underweight, 1 normal, 2 overweight, 3 obese
    out["bmi_category"] = np.digitize(out["BMI"], [18.5, 25.0, 30.0]).astype(float)
    # count of cardiovascular risk conditions
    out["cardio_risk_count"] = out[["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"]].sum(axis=1)
    # healthy habits minus unhealthy habits (range -2 .. 3)
    out["lifestyle_score"] = (out["PhysActivity"] + out["Fruits"] + out["Veggies"]
                              - out["Smoker"] - out["HvyAlcoholConsump"])
    # combined days of poor health, capped at 30
    out["poor_health_days"] = (out["MentHlth"] + out["PhysHlth"]).clip(upper=30)
    return out
