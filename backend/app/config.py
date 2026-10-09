"""
Central configuration for the Fraud Detection backend.
All values can be overridden via environment variables (see .env.example).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Database -----------------------------------------------------------
# Defaults to a local SQLite file so the API runs out-of-the-box with zero
# setup. Point DATABASE_URL at Postgres for the real deployment, e.g.:
#   postgresql+psycopg2://fraud_user:fraud_pass@localhost:5432/fraud_db
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'fraud.db'}")

# Hosts like Render/Heroku give "postgres://" or "postgresql://" URLs, but
# SQLAlchemy needs the driver name too. Fix it automatically.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# --- Model artifacts ------------------------------------------------------
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
ENCODER_PATH = ARTIFACTS_DIR / "type_encoder.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# --- Risk scoring thresholds (must match the PPT: 0-30 / 31-70 / 71-100) -
RISK_LOW_MAX = 30
RISK_MEDIUM_MAX = 70

# --- Misc ------------------------------------------------------------------
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
API_TITLE = "AI Financial Fraud Detection API"
