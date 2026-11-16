# Pilot Global Enterprise — Hyper-Personalized Specialist SDM Suite

Trained specifically on Pilot Global Enterprise's transaction history in **13.34 seconds**.
Total Model Fleet Footprint: **13553.7 KB** (4 ONNX binaries).

## Quickstart (Node.js / TypeScript)

```typescript
import { SpecialistSDMClient } from "@client-sdm/engine";

const client = new SpecialistSDMClient({
  modelsPath: "./models",
  slaTimeoutMs: 15.0
});

// Real-time evaluation in 0.08ms
const decision = await client.decide({
  amount: 85.00,
  currency: "EUR",
  card_brand: "mastercard",
  bin_country: "DE",
  velocity_1h: 1
});

console.log(decision.choice);       // "adyen_eu"
console.log(decision.confidence);   // 0.941
console.log(decision.latency_ms);    // 0.08ms
```

## Production Performance
- P99 In-Flight Latency: **0.12 ms**
- Combined Multi-Agent DAG Latency: **0.61 ms** (SLA < 20.0 ms)
- Projected Authorization Lift: **+1.25%**
