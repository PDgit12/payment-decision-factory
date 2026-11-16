# Multi-stage production container for Payment Decision Sidecar
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY sdk/sidecar/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/sdk/python
ENV SPEC_PATH=/app/specs/smart_routing.yaml
ENV PORT=8080

# Copy application code, specs, and SDK
COPY core/ /app/payment_decision_factory/core/
COPY specs/ /app/specs/
COPY sdk/ /app/payment_decision_factory/sdk/
COPY sdk/sidecar/server.py /app/server.py

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--access-log"]
