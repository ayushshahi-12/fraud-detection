from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime

from app.database import Base


class PredictionRecord(Base):
    """One row per transaction scored by the API — powers history,
    auditability and the Streamlit dashboard."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # request fields
    type = Column(String, index=True)
    amount = Column(Float)
    step = Column(Integer)
    oldbalance_org = Column(Float)
    newbalance_orig = Column(Float)
    oldbalance_dest = Column(Float)
    newbalance_dest = Column(Float)

    # model output
    fraud_probability = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String, index=True)
    prediction = Column(String)  # "Fraud" | "Legitimate"
    model_version = Column(String)
