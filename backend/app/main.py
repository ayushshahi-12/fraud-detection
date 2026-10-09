from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE
from app.database import Base, engine
from app.routers import predict, dashboard

# Create tables on startup if they don't exist yet (fine for a project this
# size — a real production system would use Alembic migrations instead).
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=API_TITLE,
    description="Serves fraud predictions + risk scores for the AI-Powered "
    "Financial Fraud Detection & Risk Scoring System project.",
    version="1.0.0",
)

# Streamlit runs on a different port/origin, so it needs CORS enabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(dashboard.router)


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}
