"""Request/response models of the REST API (the I/O specification in docs/LLD.md)."""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = ("This is a screening and awareness tool, not a medical diagnosis. "
              "Please consult a doctor for a proper check-up.")


class PatientInput(BaseModel):
    """The 21 health indicators of the CDC BRFSS survey. Binary fields: 0 = no, 1 = yes."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31.2, "Smoker": 0, "Stroke": 0,
        "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 1, "Veggies": 1,
        "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
        "MentHlth": 2, "PhysHlth": 5, "DiffWalk": 0, "Sex": 1, "Age": 9, "Education": 5,
        "Income": 6}})

    HighBP: int = Field(..., ge=0, le=1, description="Told by a doctor you have high blood pressure")
    HighChol: int = Field(..., ge=0, le=1, description="Told by a doctor you have high cholesterol")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol checked in the last 5 years")
    BMI: float = Field(..., ge=12, le=98, description="Body mass index (kg/m^2)")
    Smoker: int = Field(..., ge=0, le=1, description="Smoked at least 100 cigarettes in your life")
    Stroke: int = Field(..., ge=0, le=1, description="Ever had a stroke")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Coronary heart disease or heart attack")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity in the past 30 days (not job)")
    Fruits: int = Field(..., ge=0, le=1, description="Eat fruit 1+ times per day")
    Veggies: int = Field(..., ge=0, le=1, description="Eat vegetables 1+ times per day")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Men 14+, women 7+ drinks per week")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Have any kind of health care coverage")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Needed a doctor in last year but could not afford it")
    GenHlth: int = Field(..., ge=1, le=5, description="General health: 1 excellent ... 5 poor")
    MentHlth: int = Field(..., ge=0, le=30, description="Days of poor mental health in last 30 days")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days of poor physical health in last 30 days")
    DiffWalk: int = Field(..., ge=0, le=1, description="Serious difficulty walking or climbing stairs")
    Sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    Age: int = Field(..., ge=1, le=13, description="Age group: 1 = 18-24, 2 = 25-29, ... 13 = 80+")
    Education: int = Field(..., ge=1, le=6, description="1 = never attended ... 6 = college graduate")
    Income: int = Field(..., ge=1, le=8, description="1 = < $10k ... 8 = $75k+")


class RiskFactor(BaseModel):
    feature: str
    label: str
    current_value: float
    healthier_value: float
    impact: float = Field(..., description="Drop in risk score (0-1) if this factor were at the healthier value")


class PredictionResponse(BaseModel):
    request_id: str
    probability: float = Field(..., ge=0, le=1, description="Model risk score")
    prediction: int = Field(..., description="1 = at risk (score >= threshold), 0 = not at risk")
    risk_level: str = Field(..., description="low | moderate | high")
    threshold: float
    top_factors: List[RiskFactor]
    model_name: str
    model_version: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    inference_mode: str
    dependencies: Dict[str, bool] = {}


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: Optional[list] = None
