# Multi-stage Dockerfile for Fraud Detection API
# Stage 1: Build stage with all dependencies
# Stage 2: Runtime stage with only what's needed to serve the API

# ==================== STAGE 1: Builder ====================
FROM python:3.13-slim AS builder

WORKDIR /build

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer caching — dependencies change rarely)
COPY requirements-ci.txt .

# Install Python dependencies into a virtual environment
# (venv in builder stage keeps the final image clean)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-ci.txt

# ==================== STAGE 2: Runtime ====================
FROM python:3.13-slim

WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the API needs to run
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
# The trained model artifacts (model.ubj + MLmodel metadata). We copy ONLY
# this one model's folder — not the whole mlruns/ tree and not mlflow.db.
# Loading via MODEL_URI (below) reads straight from this path, so the image
# never depends on the registry's absolute Windows paths (which are invalid
# inside a Linux container).
COPY mlruns/1/models/m-ece3b833e50d4df796a38d89b5bf2674/artifacts/ ./model/

# Tell the app to load the model directly from the copied folder instead of
# the MLflow registry. This is what makes the image portable.
ENV MODEL_URI=/app/model

# Expose the port FastAPI will listen on
EXPOSE 8000

# Health check — Docker will ping this endpoint to verify the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the FastAPI server
# --host 0.0.0.0  : listen on all network interfaces (not just localhost)
# --port 8000     : the port we exposed above
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
