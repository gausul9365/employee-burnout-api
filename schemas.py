from enum import Enum
from pydantic import BaseModel


class Gender(str, Enum):
    male = "Male"
    female = "Female"


class JobRole(str, Enum):
    analyst = "Analyst"
    engineer = "Engineer"
    hr = "HR"
    manager = "Manager"
    sales = "Sales"


class EmployeeData(BaseModel):
    Age: float
    Gender: Gender
    Experience: float
    WorkHoursPerWeek: float
    RemoteRatio: float
    SatisfactionLevel: float
    StressLevel: float
    JobRole: JobRole


class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    probability: float
    confidence: float
    model_version: str
    timestamp: str