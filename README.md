# AI-Powered Financial Fraud Detection & Risk Scoring System

Final Year Project • B.Tech CSE • Ayush Shahi

A transaction fraud-detection system: a trained classifier scores each
transaction, converts the probability into a 0-100 risk score (LOW /
MEDIUM / HIGH), serves it through a FastAPI REST endpoint, stores every
prediction in PostgreSQL, and visualizes it all in a Streamlit dashboard.

```
User / UI  →  Streamlit  →  FastAPI  →  Preprocess  →  Model  →  Risk Level
                                              ↓
                                        PostgreSQL (history + dashboard)
```

## Project structure

```
fraud_project/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app + CORS + startup
│   │   ├── config.py        env-driven settings (DB URL, risk thresholds)
│   │   ├── database.py      SQLAlchemy engine/session
│   │   ├── models.py        PredictionRecord ORM table
│   │   ├── schemas.py       Pydantic request/response models
│   │   ├── ml_model.py      feature engineering + inference + risk scoring
│   │   ├── routers/
│   │   │   ├── predict.py   POST /predict
│   │   │   └── dashboard.py GET /stats, GET /transactions
│   │   └── artifacts/       trained model + encoder + metadata (generated)
│   ├── train_model.py       trains the classifier, prints real metrics
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py     transaction-check screen + monitoring dashboard
│   ├── requirements.txt
│   └── Dockerfile
├── data/                    put your PaySim CSV here (not included)
├── docker-compose.yml
└── .env.example
```

## Running locally (no Docker)

```bash
# 1. Backend
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py          # trains on data/paysim.csv if present,
                                # otherwise on a synthetic PaySim-shaped sample
uvicorn app.main:app --reload  # http://localhost:8000/docs

# 2. Frontend (in a second terminal)
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py   # http://localhost:8501
```

By default the backend uses a local SQLite file
(`backend/app/fraud.db`) so it runs with zero setup. Set `DATABASE_URL`
to point at Postgres for the real deployment — see `.env.example`.

## Running with Docker Compose

```bash
docker compose up --build
```

This starts Postgres, the FastAPI backend (`:8000`) and the Streamlit
dashboard (`:8501`) together, wired to talk to each other.

## Using the real PaySim dataset

Download PaySim from Kaggle, save it as `data/paysim.csv`, then:

```bash
cd backend
python train_model.py --data ../data/paysim.csv
```

The script prints precision, recall, F1-score, ROC-AUC and a confusion
matrix on a held-out test set — **use those numbers**, not invented
ones, anywhere the project reports model performance (PPT, report,
viva).

## API reference

| Method | Endpoint         | Purpose                                   |
|--------|------------------|--------------------------------------------|
| POST   | `/predict`       | Score one transaction, returns risk level  |
| GET    | `/stats`         | Aggregates for the dashboard summary cards |
| GET    | `/transactions`  | Recent prediction history                  |
| GET    | `/health`        | Liveness check                             |

Interactive docs (Swagger UI) are auto-generated at `/docs` once the
backend is running.

## Notes on the current model

`train_model.py` falls back to a synthetic, PaySim-shaped dataset when
no real CSV is found, purely so the pipeline runs end-to-end without a
470MB download. **Train on the real PaySim dataset before reporting
any accuracy/precision/recall/F1 numbers** — synthetic-data metrics are
not representative and should not appear in the report or PPT.
