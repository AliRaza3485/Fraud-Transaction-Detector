"""
schemas.py

Pydantic models that define the shape of the API's request and response.

- Transaction : what the client MUST send (validated automatically)
- PredictionResponse : what the API sends back
- HealthResponse : simple status payload for the /health endpoint

Pydantic validates incoming JSON for us: wrong types or missing fields
produce a clean 422 error instead of crashing inside the model.
"""

from enum import Enum
from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    """The 5 transaction types present in the training data."""
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class Transaction(BaseModel):
    """A single transaction to score for fraud."""

    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., ge=0, description="Transaction amount (>= 0)")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """The API's answer for one transaction."""

    fraud_probability: float = Field(..., description="Model's P(fraud), 0..1")
    is_fraud_predicted: int = Field(..., description="1 = fraud, 0 = not fraud")


class HealthResponse(BaseModel):
    """Status payload used by the health-check endpoint."""

    status: str
    model_loaded: bool
