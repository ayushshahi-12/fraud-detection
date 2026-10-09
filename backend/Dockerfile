FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If a trained model is already committed in app/artifacts, use it.
# Otherwise train a DEMO model now (data/paysim.csv is not inside this
# image, so it uses synthetic data). For real results: train locally on
# PaySim, commit backend/app/artifacts/*, and this step is skipped.
RUN [ -f app/artifacts/fraud_model.joblib ] || python train_model.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
