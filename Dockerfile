FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serialized weights, configuration variables, and code paths
COPY models/encoders.pkl models/scaler.pkl models/feature_columns.pkl models/random_forest_model.pkl models/cnn_model.keras ./models/
COPY app.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]