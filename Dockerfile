FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir jupyterlab

COPY . .

# Default: run the training pipeline
CMD ["python", "strava-moving-time-estimator-llm-rouvy.py"]
