from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Payload the client/UI sends to POST /predict."""

    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., gt=0, description="Transaction amount")
    step: int = Field(1, ge=0, description="Time step (1 step = 1 hour, as in PaySim)")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after transaction")
    oldbalanceDest: float = Field(0.0, ge=0, description="Receiver balance before transaction")
    newbalanceDest: float = Field(0.0, ge=0, description="Receiver balance after transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "TRANSFER",
                "amount": 85000,
                "step": 521,
                "oldbalanceOrg": 90000,
                "newbalanceOrig": 5000,
                "oldbalanceDest": 0,
                "newbalanceDest": 0,
            }
        }


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    prediction: Literal["Fraud", "Legitimate"]
    model_version: str
    created_at: datetime


class PredictionHistoryItem(PredictionResponse):
    type: str
    amount: float

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_transactions: int
    fraud_count: int
    fraud_rate_pct: float
    high_risk_count: int
    fraud_by_type: dict
    recent_high_risk: list[PredictionHistoryItem]
