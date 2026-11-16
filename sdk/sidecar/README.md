# Payment Decision Sidecar Microservice

The **Payment Decision Sidecar** is a containerized, sub-20ms HTTP microservice exposing real-time payment gateway routing and risk decisioning.

## Architecture

```
[Merchant Checkout API]
         │
         ├── (HTTP keep-alive, <5ms loopback) ──► [Decision Sidecar :8080]
         │                                              │
         ▼                                              ▼ (In-Memory Engine)
 [Selected Gateway: Stripe / Adyen / Checkout.com]  [Calibrated Spec v1]
```

## Endpoints

| Endpoint | Method | Description | Target Latency |
|---|---|---|---|
| `/decide` | POST | Single payment context routing evaluation | < 20ms SLA (< 2ms local) |
| `/decide/batch` | POST | High-volume batch transaction evaluation | < 50ms per 100 tx |
| `/health` | GET | Liveness and model readiness probe | < 1ms |
| `/metrics` | GET | Throughput, latency percentiles, and gateway split | < 1ms |

## Running Locally

### 1. Direct Python Execution
```bash
python server.py
# Server runs on http://0.0.0.0:8080
```

### 2. Using Docker Compose
```bash
docker compose up --build -d
```

### 3. Verify Health
```bash
curl http://localhost:8080/health
```

### 4. Test Single Decision
```bash
curl -X POST http://localhost:8080/decide \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 149.99,
    "currency": "EUR",
    "card_brand": "mastercard",
    "bin_country": "DE",
    "mcc": "5411",
    "velocity_1h": 1
  }'
```
