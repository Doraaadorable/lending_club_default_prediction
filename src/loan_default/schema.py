"""What a valid loan application looks like, shared by the API and by training.

The API validates each request with `LoanApplication`. Training runs `check_frame` on
the training and test files, so both sides hold the same invariants.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

EmploymentLength = Literal[
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
    "NI",
]


class LoanApplication(BaseModel):
    """Application-time information about one loan. Field names match the training data."""

    revenue: float = Field(gt=0, description="Self-reported annual income, USD")
    loan_amnt: float = Field(gt=0, description="Requested loan amount, USD")
    fico_n: float = Field(ge=300, le=850, description="FICO score (midpoint of the reported range)")
    dti_n: float | None = Field(default=None, ge=0, description="Debt-to-income ratio, percent; omit if unknown")
    emp_length: EmploymentLength = Field(default="NI", description="Employment length; NI means no information")
    purpose: str = Field(min_length=1, description="Loan purpose, e.g. debt_consolidation, credit_card, car")
    home_ownership_n: str = Field(min_length=1, description="RENT, MORTGAGE, OWN or OTHER")
    addr_state: str = Field(pattern="^[A-Z]{2}$", description="Two-letter US state code")
    application_id: str | None = Field(
        default=None,
        max_length=64,
        description="Your reference for the application; logged so outcomes can be joined to predictions later",
    )


# The fields the model reads; application_id is only carried into the prediction log.
FIELDS = [name for name in LoanApplication.model_fields if name != "application_id"]
_many = TypeAdapter(list[LoanApplication])


def check_frame(df: pd.DataFrame, name: str, max_errors: int = 5) -> None:
    """Raise ValueError if any row breaks the application schema.

    Missing values are allowed where the API allows them to be omitted (`dti_n`), so
    NaN is passed through as None. Codes the cleaning step handles, such as the DTI
    placeholder 999, are valid input.
    """
    missing = [field for field in FIELDS if field not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
    records = df[FIELDS].astype(object).where(df[FIELDS].notna(), None).to_dict("records")
    try:
        _many.validate_python(records)
    except ValidationError as exc:
        problems = [f"row {err['loc'][0]}, {err['loc'][1]}: {err['msg']}" for err in exc.errors()[:max_errors]]
        raise ValueError(
            f"{name} breaks the application schema ({exc.error_count()} errors): " + "; ".join(problems)
        ) from exc
