# ====================================================================
# VoiceRAG: Production Cloud Dockerfile
# Optimized for Sub-200ms Low-Latency Inference & Zero Cold-Starts
# ====================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Set environment variables for production execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application source code
COPY . .

# Expose standard web port
EXPOSE 8000

# Healthcheck for zero-downtime rolling cloud deployments
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Launch high-performance Uvicorn server with 4 asynchronous workers
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "auto"]
