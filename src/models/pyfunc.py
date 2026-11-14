"""MLflow pyfunc wrapper: the deployable model served by `mlflow models serve`.

Input : DataFrame with the 21 raw CDC features (any column order).
Output: DataFrame with probability, prediction (0/1), risk_level and model_version per row.
"""
import mlflow.pyfunc
import pandas as pd

from src.common.schema import FEATURE_NAMES, risk_level


class DiabetesRiskPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        bundle = joblib.load(context.artifacts["bundle"])
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        self.version = bundle["model_version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[FEATURE_NAMES].astype(float)
        proba = self.pipeline.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "probability": proba.round(4),
            "prediction": (proba >= self.threshold).astype(int),
            "risk_level": [risk_level(p, self.threshold) for p in proba],
            "model_version": self.version,
        })
