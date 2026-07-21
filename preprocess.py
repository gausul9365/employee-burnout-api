import torch

from schemas import EmployeeData, Gender, JobRole


def prepare_input(employee: EmployeeData):

    # -------------------------
    # Gender Encoding
    # -------------------------

    gender = 0 if employee.Gender == Gender.male else 1

    # -------------------------
    # JobRole Encoding
    # -------------------------

    analyst = 1 if employee.JobRole == JobRole.analyst else 0
    engineer = 1 if employee.JobRole == JobRole.engineer else 0
    hr = 1 if employee.JobRole == JobRole.hr else 0
    manager = 1 if employee.JobRole == JobRole.manager else 0
    sales = 1 if employee.JobRole == JobRole.sales else 0

    # -------------------------
    # Feature Vector
    # -------------------------

    features = [
        employee.Age,
        gender,
        employee.Experience,
        employee.WorkHoursPerWeek,
        employee.RemoteRatio,
        employee.SatisfactionLevel,
        employee.StressLevel,
        analyst,
        engineer,
        hr,
        manager,
        sales
    ]

    return torch.tensor(
        [features],
        dtype=torch.float32
    )