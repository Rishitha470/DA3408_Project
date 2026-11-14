"""Single source of truth for the input features of the CDC Diabetes Health Indicators dataset.

Used by the data pipeline (validation), the training code (feature order), the backend
(request validation and explanations) and the drift monitor.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

TARGET = "Diabetes_binary"


@dataclass(frozen=True)
class Feature:
    """Metadata for one input feature."""

    name: str
    label: str              # plain-language name shown to users
    min: float
    max: float
    kind: str               # "binary", "ordinal" or "numeric"
    healthy: Optional[float] = None   # reference value used for "what-if" explanations
    modifiable: bool = False          # can the person change it through lifestyle / treatment?


FEATURES: List[Feature] = [
    Feature("HighBP", "High blood pressure", 0, 1, "binary", 0, True),
    Feature("HighChol", "High cholesterol", 0, 1, "binary", 0, True),
    Feature("CholCheck", "Cholesterol check in last 5 years", 0, 1, "binary"),
    Feature("BMI", "Body mass index (BMI)", 12, 98, "numeric", 23, True),
    Feature("Smoker", "Smoked 100+ cigarettes in lifetime", 0, 1, "binary"),
    Feature("Stroke", "Ever had a stroke", 0, 1, "binary"),
    Feature("HeartDiseaseorAttack", "Heart disease or heart attack", 0, 1, "binary"),
    Feature("PhysActivity", "Physical activity in last 30 days", 0, 1, "binary", 1, True),
    Feature("Fruits", "Eats fruit daily", 0, 1, "binary", 1, True),
    Feature("Veggies", "Eats vegetables daily", 0, 1, "binary", 1, True),
    Feature("HvyAlcoholConsump", "Heavy alcohol consumption", 0, 1, "binary", 0, True),
    Feature("AnyHealthcare", "Has any health coverage", 0, 1, "binary"),
    Feature("NoDocbcCost", "Could not see a doctor because of cost", 0, 1, "binary"),
    Feature("GenHlth", "General health (1 = excellent, 5 = poor)", 1, 5, "ordinal", 2, True),
    Feature("MentHlth", "Days of poor mental health (last 30)", 0, 30, "numeric", 0, True),
    Feature("PhysHlth", "Days of poor physical health (last 30)", 0, 30, "numeric", 0, True),
    Feature("DiffWalk", "Serious difficulty walking or climbing stairs", 0, 1, "binary"),
    Feature("Sex", "Sex (0 = female, 1 = male)", 0, 1, "binary"),
    Feature("Age", "Age group (1 = 18-24 ... 13 = 80+)", 1, 13, "ordinal"),
    Feature("Education", "Education level (1-6)", 1, 6, "ordinal"),
    Feature("Income", "Income level (1-8)", 1, 8, "ordinal"),
]

FEATURE_NAMES: List[str] = [f.name for f in FEATURES]
FEATURE_BY_NAME: Dict[str, Feature] = {f.name: f for f in FEATURES}
ALL_COLUMNS: List[str] = FEATURE_NAMES + [TARGET]

# A score at or above the decision threshold is "high"; between MODERATE_FRACTION * threshold
# and the threshold is "moderate"; below that is "low". Keeping the bands tied to the tuned
# threshold means the risk level always agrees with the model's yes/no prediction.
MODERATE_FRACTION = 0.6


def risk_level(probability: float, threshold: float) -> str:
    """Map a model score to 'low', 'moderate' or 'high' relative to the decision threshold."""
    if probability >= threshold:
        return "high"
    if probability >= MODERATE_FRACTION * threshold:
        return "moderate"
    return "low"
