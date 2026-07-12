"""Pydantic request/response models: strict validation at the service boundary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """Raw customer attributes — exactly what the training data contains.

    Feature derivation happens server-side via the same build_features()
    used in training, so clients send raw fields only.
    """

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0, le=120)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0, le=200000)


class ScoringResponse(BaseModel):
    """The service returns a business decision, not just a raw score."""

    probability: float = Field(ge=0, le=1, description="Calibrated churn probability")
    decision: Literal["call", "no_call"]
    threshold: float = Field(description="Decision threshold derived from unit economics")
    expected_value_of_call: float = Field(description="EV of calling this customer, RUB: P*p*V - C")
    model_version: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[ScoringResponse]
    n_calls: int


class HealthResponse(BaseModel):
    status: Literal["ok", "no_model"]
    model_version: str | None
    model_source: str | None
