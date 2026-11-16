# Payment Decision Factory - Client SDKs & Integration Guide

The **Payment Decision SDK Suite** provides ultra-low latency (<20ms SLA guaranteed, <1ms in-process) intelligent decisioning and gateway dispatch modules for high-conversion fintech platforms, checkout flows, and payment processors.

```
                      ┌──────────────────────────────────────────────┐
                      │            Client Checkout Request           │
                      └──────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │     Python In-Process     │               │     Sidecar Container     │
         │         (< 1ms)           │               │       (< 5ms loopback)    │
         │   `payment_decision_sdk`  │               │      `POST /decide`       │
         └─────────────┬─────────────┘               └─────────────┬─────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │    Calibrated Routing Decision Output        │
                      │  Choice | Confidence | Attribution | Noul    │
                      └──────────────────────┬───────────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
  ┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
  │      Stripe US      │         │       Adyen EU      │         │    Checkout.com     │
  │ Domestic US & Amex  │         │  Intra-EU & PSD2    │         │ Multi-Currency FX   │
  └─────────────────────┘         └─────────────────────┘         └─────────────────────┘
```

---

## Latency SLA Profiles

| Integration Mode | Latency (p50) | Latency (p99) | SLA Budget | Target Use Case |
|---|---|---|---|---|
| **Python In-Process** | `0.06 ms` | `0.18 ms` | `< 1.0 ms` | Core backend checkout services written in Python |
| **Node / TS In-Process** | `0.11 ms` | `0.45 ms` | `< 1.0 ms` | Edge workers, Next.js / Express checkout APIs |
| **Local Sidecar (HTTP)** | `1.2 ms` | `3.8 ms` | `< 20.0 ms` | Polyglot microservices (Go, Java, Ruby, Rust, PHP) |
| **Sidecar Fallback** | `0.02 ms` | `0.08 ms` | `< 20.0 ms` | Automatic zero-dropout safety circuit breaker |

---

## 1. Python Client SDK (`payment_decision_sdk`)

### Installation

```bash
cd payment_decision_factory/sdk/python
pip install -e .
```

### Synchronous In-Process Routing (< 1ms)

```python
from payment_decision_sdk import DecisionEngineClient, PaymentContext

# Initialize client in ultra-low latency in-process mode
client = DecisionEngineClient(mode="in_process")

context = PaymentContext(
    amount=149.50,
    currency="USD",
    card_brand="visa",
    bin_country="US",
    mcc="5411",
    velocity_1h=1,
    customer_id="cust_9876",
)

result = client.decide(context)

print(f"Optimal Gateway Rail : {result.decision}")
print(f"Confidence Score     : {result.confidence * 100:.1f}%")
print(f"Inference Latency    : {result.latency_ms:.3f} ms")
print(f"Routing Rationale    : {result.routing_reason}")
```

### Asynchronous Checkout Pipeline

```python
import asyncio
from payment_decision_sdk import DecisionEngineClient

async def process_payment(tx: dict):
    async with DecisionEngineClient(mode="in_process") as client:
        decision = await client.decide_async(tx)
        
        if decision.decision == "stripe_us":
            return await dispatch_to_stripe(tx)
        elif decision.decision == "adyen_eu":
            return await dispatch_to_adyen(tx)
        else:
            return await dispatch_to_checkout(tx)

asyncio.run(process_payment({
    "amount": 250.0,
    "currency": "EUR",
    "card_brand": "mastercard",
    "bin_country": "DE",
    "mcc": "5732",
}))
```

### Batch Evaluation for Reconciliation & Settlements

```python
transactions = [
    {"amount": 45.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411"},
    {"amount": 89.0, "currency": "GBP", "card_brand": "mastercard", "bin_country": "GB", "mcc": "5411"},
    {"amount": 1200.0, "currency": "JPY", "card_brand": "jcb", "bin_country": "JP", "mcc": "5732"},
]

results = client.decide_batch(transactions)
for r in results:
    print(f"{r.decision} | confidence: {r.confidence:.2f} | {r.latency_ms:.3f}ms")
```

---

## 2. TypeScript / Node.js Client

### Installation & Setup

```bash
cd payment_decision_factory/sdk/typescript
npm install
```

### Pre-Gateway Dispatch in Checkout Route

```typescript
import { PaymentDecisionClient } from "@payment-decision-factory/client";
import type { PaymentContext } from "@payment-decision-factory/client";

// Initialize client with strict 20ms SLA latency budget
const client = new PaymentDecisionClient({
  sidecarUrl: "http://localhost:8080",
  timeoutMs: 20,              // Strict SLA budget
  enableLocalFallback: true,  // Automatically activates zero-dropout fallback
});

export async function handleCheckout(req, res) {
  const context: PaymentContext = {
    amount: req.body.amount,
    currency: req.body.currency,
    card_brand: req.body.cardBrand,
    bin_country: req.body.binCountry,
    mcc: req.body.mcc,
    velocity_1h: req.body.userHourlyAttempts ?? 1,
  };

  // Evaluate optimal gateway within 20ms budget
  const decision = await client.decide(context);

  console.log(`Routed to: ${decision.decision} in ${decision.latency_ms}ms`);

  // Dispatch payment
  switch (decision.decision) {
    case "stripe_us":
      return await stripeGateway.charge(req.body);
    case "adyen_eu":
      return await adyenGateway.processPayment(req.body);
    default:
      return await checkoutComGateway.authorize(req.body);
  }
}
```

### Zero-Dropout Fallback Architecture

If the decision sidecar container is restarting during rolling deployments or experiencing network jitter exceeding the configured 20ms SLA timeout:
1. `AbortController` triggers at `timeoutMs`.
2. The TypeScript client **instantly falls back** to its embedded calibrated scoring model in `<0.1ms`.
3. The checkout proceeds with the intelligent fallback gateway, setting `is_fallback: true` and logging a telemetry alert.
4. **Checkout revenue conversion is 100% protected—no payment request is ever dropped.**

---

## 3. Sidecar Microservice Container (`sdk/sidecar/`)

The sidecar container package exposes a high-throughput, low-latency REST API designed to run as a sidecar container in Kubernetes or Docker networks.

### Quick Start with Docker Compose

```bash
cd payment_decision_factory/sdk/sidecar
docker compose up --build -d
```

### Health & Readiness Check

```bash
curl -i http://localhost:8080/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "spec_id": "smart_routing_v1",
  "spec_version": "1.0.0",
  "uptime_sec": 48.2
}
```

### Single Decision Endpoint (`POST /decide`)

```bash
curl -X POST http://localhost:8080/decide \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 79.99,
    "currency": "EUR",
    "card_brand": "mastercard",
    "bin_country": "FR",
    "mcc": "5411",
    "velocity_1h": 1
  }'
```

**Response:**
```json
{
  "decision": "adyen_eu",
  "choice": "adyen_eu",
  "confidence": 0.5757,
  "probabilities": {
    "stripe_us": 0.2281,
    "adyen_eu": 0.5757,
    "checkout_crossborder": 0.1962
  },
  "scores": {
    "stripe_us": 0.05,
    "adyen_eu": 0.85,
    "checkout_crossborder": 0.1
  },
  "feature_contributions": {
    "geo_affinity": 0.4,
    "currency_affinity": 0.35,
    "brand_affinity": 0.15
  },
  "latency_ms": 1.45,
  "routing_reason": "Optimal match: Adyen European Direct Clearing Rail selected with 57.6% confidence based on INTRA_EU corridor and currency alignment (EUR).",
  "is_fallback": false,
  "noul_triggered": false,
  "noul_action": null,
  "effective_fee_est": 128.5,
  "model_version": "1.0.0",
  "timestamp": 1731640120.45
}
```

### High-Volume Batch Endpoint (`POST /decide/batch`)

```bash
curl -X POST http://localhost:8080/decide/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": 100.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411"},
      {"amount": 250.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "DE", "mcc": "5411"}
    ]
  }'
```

### Metrics & Telemetry (`GET /metrics`)

```bash
curl http://localhost:8080/metrics
```
```json
{
  "total_decisions": 1420,
  "total_batch_transactions": 850,
  "average_inference_latency_ms": 0.284,
  "circuit_breakers_triggered": 3,
  "decisions_by_gateway": {
    "stripe_us": 840,
    "adyen_eu": 430,
    "checkout_crossborder": 150
  },
  "uptime_seconds": 1240.5
}
```

---

## 4. Specialist Small Decision Model (SDM) Suite

Instead of deploying heavy, slow 800MB LLMs into checkout pathways, the platform provides a suite of **Specialist Small Decision Models (SDMs)** (<2MB total bundle, <0.1ms inference, trained in <30s from scratch on client telemetry):
- **Routing Specialist SDM**: Multi-corridor cost, acceptance, and interchange optimization.
- **Fraud Risk Specialist SDM**: Pre-auth card-testing burst and velocity risk scoring.
- **Dynamic 3DS / SCA Specialist SDM**: PSD2 RTS Article 16/18 frictionless exemptions.
- **Smart Retry Specialist SDM**: Scheme fine-prevention and liquidity-aligned retry scheduling.

### Python SDK Specialist SDM Evaluation

```python
from payment_decision_sdk import DecisionEngineClient, SpecialistSDMEvaluator

# Option A: Standalone Specialist Evaluator (ONNX, PyTorch, or Native)
evaluator = SpecialistSDMEvaluator(specialist_type="routing")
res = evaluator.evaluate({
    "amount": 125.0,
    "currency": "EUR",
    "card_brand": "mastercard",
    "bin_country": "DE",
    "mcc": "5411",
    "velocity_1h": 1,
})
print(f"SDM Decision: {res.decision} (p={res.confidence:.3f}) in {res.latency_ms:.3f}ms")

# Option B: Via DecisionEngineClient
client = DecisionEngineClient(mode="sdm", specialist_type="fraud")
fraud_res = client.decide({"amount": 50.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411", "velocity_1h": 15})
print(f"Risk Assessment: {fraud_res.decision}")
```

### TypeScript / Node.js Specialist SDM Evaluation

```typescript
import { SpecialistSDMClient } from "@payment-decision-factory/client";

// In-process microsecond evaluation (<0.08ms)
const routingSDM = new SpecialistSDMClient({ specialistType: "routing", inProcess: true });
const decision = routingSDM.decideSync({
  amount: 99.0,
  currency: "USD",
  card_brand: "amex",
  bin_country: "US",
  mcc: "5411",
  velocity_1h: 1,
});

console.log(`Routed to: ${decision.decision} with ${(decision.confidence * 100).toFixed(1)}% confidence`);

// 3DS / SCA Exemption evaluation
const scaSDM = new SpecialistSDMClient({ specialistType: "sca", inProcess: true });
const scaDecision = scaSDM.decideSync({
  amount: 22.50, // Low value < €30
  currency: "EUR",
  card_brand: "visa",
  bin_country: "FR",
  mcc: "5411",
});
console.log(`SCA Exemption: ${scaDecision.decision}`); // frictionless_low_value
```

### Sidecar REST API for Specialist SDMs

```bash
# Evaluate via dedicated specialist endpoint
curl -X POST http://localhost:8080/specialists/decide \
  -H "Content-Type: application/json" \
  -d '{
    "specialist_type": "routing",
    "context": {
      "amount": 110.0,
      "currency": "EUR",
      "card_brand": "mastercard",
      "bin_country": "DE",
      "mcc": "5411",
      "velocity_1h": 1
    }
  }'
```

---

## 5. Sealed Closed-Source Distribution (`.sdmpkg`)

For enterprise environments requiring source IP protection and client license entitlement, models are compiled into tamper-evident **Sealed Packages (`.sdmpkg`)**:
- **Encryption**: Authenticated AES-256-GCM with 256-bit keys derived via PBKDF2-HMAC-SHA256 (100,000 iterations).
- **Tamper-Evident Binding**: Package manifest (client ID, expiration timestamp, permissions) bound as AAD; any file modification invalidates the cryptographic authentication tag.
- **Zero-Disk Plaintext Leakage**: Models decrypt strictly into host RAM and execute directly in-memory (<0.1ms).

### Python SDK Sealed Package Loading

```python
from payment_decision_sdk import DecisionEngineClient, SealedPackageClient

# Option A: Direct Sealed Package Client (supports compact & enterprise sub-billion tiers)
client = SealedPackageClient(
    package_path="artifacts/specialist_model.sdmpkg",
    license_key="enterprise-secret-license-key",
    client_id="acme_corp",
)
print(f"Loaded tier: {client.tier} (Enterprise: {client.is_enterprise_tier})")
result = client.decide({"amount": 85.0, "currency": "USD", "card_brand": "visa", "bin_country": "US", "mcc": "5411"})
print(f"Decided: {result.decision} in {result.latency_ms:.3f}ms (SLA budget: <10ms)")

# Option B: Unified DecisionEngineClient
engine = DecisionEngineClient(
    mode="sealed",
    sealed_package_path="artifacts/specialist_model.sdmpkg",
    license_key="enterprise-secret-license-key",
    client_id="acme_corp",
)
res = engine.decide({"amount": 140.0, "currency": "EUR", "card_brand": "mastercard", "bin_country": "DE", "mcc": "5411"})
```

### TypeScript Sealed Package Loading

```typescript
import { SealedSDMPackageClient } from "@payment-decision-factory/client";

const sealedClient = new SealedSDMPackageClient({
  packagePath: "artifacts/specialist_model.sdmpkg",
  licenseKey: "enterprise-secret-license-key",
  clientId: "acme_corp",
});

console.log(`Package Tier: ${sealedClient.tier} | Enterprise: ${sealedClient.isEnterpriseTier}`);

const decision = sealedClient.decideSync({
  amount: 95.0,
  currency: "USD",
  card_brand: "amex",
  bin_country: "US",
  mcc: "5411",
});

console.log(`Decision: ${decision.decision} in ${decision.latency_ms}ms (p=${(decision.confidence * 100).toFixed(1)}%)`);
```

---

## 6. Enterprise Sub-Billion Parameter SDM Tier (`tier="sub_billion"`)

For global tier-1 banks, payment facilitators, and high-volume merchant platforms processing billions in GMV, the factory compiles **Enterprise Sub-Billion Parameter Specialist SDMs** (100M–500M parameters, FT-Transformer architecture):

- **Architecture**: Tabular Feature-Tokenizer Transformer with continuous numerical projections, high-cardinality categorical entity embeddings (BINs, MCCs, corridors), and multi-head self-attention layers.
- **Latency Guarantee**: Strictly optimized with sub-10ms P99 latency on standard CPU instances (typically 0.6ms–1.5ms).
- **In-Memory Encrypted Decryption**: Compiles into `.sdmpkg` archives decrypting in RAM with zero-disk plaintext exposure.
- **Readiness Certification**: Verified against Digital Twin Sandbox stress testing with counterfactual lift analysis, noise perturbation robustness, and automated policy self-adjustment.

---

## 7. Gateway Decision Matrix

| Dimension | `stripe_us` | `adyen_eu` | `checkout_crossborder` |
|---|---|---|---|
| **Primary Region** | United States & Canada | European Economic Area & UK | Global / Cross-Border Corridors |
| **Preferred Currencies** | USD, CAD | EUR, GBP, CHF, SEK, NOK | USD, EUR, GBP, AUD, JPY, SGD |
| **Card Network Affinity** | Amex, Discover, US Visa/MC | European Visa, Mastercard | JCB, UnionPay, Multi-currency cards |
| **Specialties** | Instant ACH, domestic interchange | PSD2 SCA exemptions, SEPA | FX hedging, high-ticket cross-border |
| **Expected Provider Fee** | 290 bps + $0.30 | 120 bps + €0.10 | 240 bps + $0.20 |

---

## 8. Verification & Testing

Both SDKs include complete test suites:

```bash
# Python SDK Test Suite (23 tests including enterprise sub-billion tier)
cd payment_decision_factory && python3 -P -m pytest sdk/python/tests/test_sdk.py -v

# TypeScript Client Test Suite (14 tests including enterprise sub-billion tier)
node --experimental-strip-types --test payment_decision_factory/sdk/typescript/test/client.test.ts


# TypeScript Checkout Example Walkthrough
node --experimental-strip-types payment_decision_factory/sdk/typescript/examples/checkout_example.ts
```
