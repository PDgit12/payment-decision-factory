# Payment Decision Factory — Client Sandbox & Traffic Playground

An interactive, personalized evaluation sandbox and synthetic traffic generator for **System 1 Payment Decision Models**.

The sandbox allows fintech developers, risk engineers, and payment architects to explore declarative decision policies, test parameter sensitivities, inspect sub-20ms latency execution profiles, review explainability feature attributions, and stress-test decision policies against realistic transaction traffic batches.

---

## Key Features

1. **Interactive Real-Time Parameter Tuning**:
   - Dynamic controls for **Amount** (slider & numeric input), **Currency** (USD, EUR, GBP, CAD, AUD, JPY, SGD, etc.), **Card Brand** (Visa, Mastercard, Amex, Discover, UnionPay, JCB), **BIN Issuing Country** (US, GB, DE, FR, NL, SG, etc.), **Merchant MCC**, and **Hourly Card Velocity**.
   - Real-time instant inference debounced at 120ms without page reloads.

2. **System 1 Real-Time Telemetry & SLA Tracking**:
   - **Primary Action Banner**: Shows selected clearing rail (e.g., Stripe US, Adyen EU, Checkout.com), regional domain, clearing fee estimation (bps + fixed cents), and settlement days.
   - **Sub-20ms Latency Profile**: Displays exact inference duration (typically 0.1ms – 0.5ms) against the `< 20ms` SLA target.
   - **Calibrated Confidence**: Temperature-scaled probability score ($T=0.85$) indicating model certainty.
   - **Candidate Probability Comparison**: Visual distribution across all candidate rails.
   - **Explainability (Feature Contributions)**: Attribution weights highlighting how currency match, geographic corridor, card brand, and velocity drove the choice.
   - **Noul Circuit Breakers**: Visual alert banner triggered on velocity spikes ($\ge 15\text{ tx/hr}$) or anomalous fraud vectors.

3. **Mock Payment Traffic Generator (`mock_traffic.py`)**:
   - **Happy Path**: Standard domestic and international retail e-commerce traffic ($15 – $250, velocity 1–3, balanced currency/country pairs).
   - **Card-Testing Attacks**: High-velocity automated card enumeration attacks (micro-amounts $0.50 – $5.00, velocity 15–60/hr, disparate BINs).
   - **Cross-Border PSD2**: European EEA transactions subject to PSD2 Strong Customer Authentication ($> €30$, requiring Adyen EU direct acquiring optimization).
   - **Mixed Blend**: Realistic blended workload for concurrency verification.

4. **Specialist SDM Suite Controls (<2MB, Sub-0.2ms)**:
   - **Compact Deep Neural Models**: Instant training of the 4 Specialist SDMs (Routing, Fraud Risk, Dynamic 3DS, Smart Retry) directly from telemetry (<15 seconds on CPU).
   - **Live Performance & Calibration Dashboard**: Real-time validation accuracy, Brier scores ($<0.05$), loss curves, and parameter counts (~75k parameters, ~300KB per specialist).
   - **Architectural Efficiency Comparison**: Live metrics comparing heavy 800MB LLMs ($450\text{ms}$, uncalibrated) vs compact Specialist SDMs ($0.04\text{ms}$ CPU inference, $11,250\times$ speedup, $2,800\times$ memory reduction).
   - **Playground Mode Switch**: Dynamic toggle between declarative heuristic policy and deep neural Specialist SDM inference.
   - **ONNX Export Support**: One-click export to `.onnx` for C++, Node.js, Go, or Python microsecond serving.

5. **Multi-Spec Policy Switcher**:
   - Hot-reload between different decision specs (`smart_routing.yaml`, `fraud_screener.yaml`, `dynamic_3ds.yaml`, `smart_retry.yaml`) straight from the UI.

---

## Quickstart

### 1. Launch the Sandbox Web App
```bash
python payment_decision_factory/sandbox/app.py --port 8000
```
or via Uvicorn:
```bash
uvicorn payment_decision_factory.sandbox.app:app --host 0.0.0.0 --port 8000
```

Open your browser at [http://localhost:8000](http://localhost:8000).

### 2. Run the Mock Payment Traffic Generator via CLI
```bash
# Happy path domestic/international batch
python -m payment_decision_factory.sandbox.mock_traffic --scenario happy_path --count 50

# Card-testing enumeration attack batch
python -m payment_decision_factory.sandbox.mock_traffic --scenario card_testing --count 30

# European cross-border PSD2 batch
python -m payment_decision_factory.sandbox.mock_traffic --scenario cross_border_psd2 --count 25

# Blended workload
python -m payment_decision_factory.sandbox.mock_traffic --scenario mixed --count 100
```

### 3. Run the Test Suite
```bash
cd payment_decision_factory
python -m pytest tests/test_sandbox.py -v
```

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive Client Sandbox Single-Page Application (HTML/Tailwind/JS) |
| `GET` | `/health` | Health status, active spec, engine readiness, and SLA target |
| `GET` | `/api/spec` | Full JSON schema, choices, SLAs, and primitives of active spec |
| `GET` | `/api/specs/list` | Lists all `.yaml` specs available in `specs/` |
| `POST` | `/api/specs/switch` | Dynamically switches active decision policy |
| `POST` | `/api/decide` | Evaluates a single payment transaction in real time |
| `POST` | `/api/traffic/generate` | Generates and evaluates a batch of synthetic transactions |
| `GET` | `/api/traffic/stream` | Generates a burst of evaluated transactions for live animation |
| `POST` | `/api/benchmark` | Runs an in-process 100-sample latency benchmark |
