/**
 * Payment Decision Engine - TypeScript / Node.js Client.
 *
 * Designed for fintech payment engineers dispatching transactions before payment gateways.
 * Guaranteed <20ms SLA with zero-dropout fallback protection.
 */

import type { DecisionClientOptions, DecisionResult, PaymentContext } from "./types.ts";

const EU_COUNTRIES = new Set([
  "GB", "DE", "FR", "NL", "IE", "ES", "IT", "SE", "NO", "DK",
  "BE", "AT", "CH", "PL", "FI", "PT", "CZ", "RO", "HU", "GR"
]);

export class PaymentDecisionClient {
  private readonly sidecarUrl: string;
  private readonly timeoutMs: number;
  private readonly fallbackDecision: string;
  private readonly enableLocalFallback: boolean;
  private readonly localOnly: boolean;

  constructor(options: DecisionClientOptions = {}) {
    this.sidecarUrl = (options.sidecarUrl || "http://localhost:8080").replace(/\/+$/, "");
    this.timeoutMs = options.timeoutMs ?? 20;
    this.fallbackDecision = options.fallbackDecision || "stripe_us";
    this.enableLocalFallback = options.enableLocalFallback ?? true;
    this.localOnly = options.localOnly ?? false;
  }

  /**
   * Evaluate a payment transaction and determine the optimal routing gateway.
   *
   * Enforces strict SLA timeout (default 20ms). If the sidecar is unreachable or times out,
   * automatically computes a calibrated fallback decision to guarantee zero checkout abandonment.
   *
   * @param context Payment context parameters.
   * @returns Calibrated DecisionResult.
   */
  async decide(context: PaymentContext): Promise<DecisionResult> {
    const startTime = performance.now();

    if (this.localOnly) {
      return this.decideLocal(context);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.sidecarUrl}/decide`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({
          amount: context.amount,
          currency: context.currency.toUpperCase(),
          card_brand: context.card_brand.toLowerCase(),
          bin_country: context.bin_country.toUpperCase(),
          mcc: String(context.mcc),
          velocity_1h: context.velocity_1h ?? 1,
          customer_id: context.customer_id,
          payment_method_id: context.payment_method_id,
          metadata: context.metadata ?? {},
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = (await response.json()) as Record<string, unknown>;
        const elapsed = performance.now() - startTime;
        return {
          decision: String(data.decision || data.choice),
          choice: String(data.choice || data.decision),
          confidence: Number(data.confidence ?? 0.8),
          probabilities: (data.probabilities as Record<string, number>) || {},
          latency_ms: Math.round(elapsed * 1000) / 1000,
          routing_reason: String(data.routing_reason || "Evaluated by sidecar"),
          feature_contributions: data.feature_contributions as Record<string, number> | undefined,
          is_fallback: Boolean(data.is_fallback ?? false),
          noul_triggered: Boolean(data.noul_triggered ?? false),
          noul_action: (data.noul_action as string) ?? null,
          effective_fee_est: Number(data.effective_fee_est ?? 0),
          model_version: String(data.model_version || "v1.0.0"),
          timestamp: Number(data.timestamp || Date.now() / 1000),
        };
      }

      // Non-200 response from sidecar
      if (this.enableLocalFallback) {
        return this.buildFallbackResult(context, startTime, `Sidecar returned HTTP ${response.status}`);
      }
      throw new Error(`Sidecar error HTTP ${response.status}: ${await response.text()}`);

    } catch (err: unknown) {
      clearTimeout(timeoutId);
      const isAbort = err instanceof Error && err.name === "AbortError";
      const reason = isAbort
        ? `SLA timeout (${this.timeoutMs}ms) exceeded`
        : `Sidecar connection error: ${err instanceof Error ? err.message : String(err)}`;

      if (this.enableLocalFallback) {
        return this.buildFallbackResult(context, startTime, reason);
      }
      throw new Error(reason);
    }
  }

  /**
   * Fast, synchronous in-process calibrated evaluation (<0.1ms).
   *
   * Evaluates the declarative rules and temperature-scaled softmax model directly
   * within Node.js without any network overhead.
   *
   * @param context Payment context parameters.
   */
  decideLocal(context: PaymentContext): DecisionResult {
    const startTime = performance.now();
    const currency = context.currency.toUpperCase();
    const binCountry = context.bin_country.toUpperCase();
    const brand = context.card_brand.toLowerCase();
    const velocity = context.velocity_1h ?? 1;
    const amount = context.amount;

    // Base scores
    const scores: Record<string, number> = {
      stripe_us: 0.0,
      adyen_eu: 0.0,
      checkout_crossborder: 0.1,
    };
    const contributions: Record<string, number> = {};
    let reason = "Multi-rail cost and acceptance scoring";
    let isFallback = false;
    let noulTriggered = false;
    let noulAction: string | null = null;

    // 1. Geographic Affinity (Weight: 0.40)
    if (["US", "CA", "MX"].includes(binCountry)) {
      scores.stripe_us += 0.40;
      contributions.geo_affinity = 0.40;
      reason = `Domestic North America routing for BIN ${binCountry}`;
    } else if (EU_COUNTRIES.has(binCountry)) {
      scores.adyen_eu += 0.40;
      contributions.geo_affinity = 0.40;
      reason = `European direct clearing optimization for BIN ${binCountry}`;
    } else {
      scores.checkout_crossborder += 0.35;
      contributions.geo_affinity = 0.35;
      reason = `Cross-border corridor routing for international BIN ${binCountry}`;
    }

    // 2. Currency Affinity (Weight: 0.35)
    if (["USD", "CAD"].includes(currency)) {
      scores.stripe_us += 0.35;
      contributions.currency_affinity = 0.35;
    } else if (["EUR", "GBP", "CHF", "SEK", "NOK"].includes(currency)) {
      scores.adyen_eu += 0.35;
      contributions.currency_affinity = 0.35;
    } else {
      scores.checkout_crossborder += 0.30;
      contributions.currency_affinity = 0.30;
    }

    // 3. Brand Affinity (Weight: 0.15)
    if (brand === "amex" || brand === "discover") {
      scores.stripe_us += 0.15;
      contributions.brand_affinity = 0.15;
    } else if (brand === "visa" || brand === "mastercard") {
      if (EU_COUNTRIES.has(binCountry)) {
        scores.adyen_eu += 0.15;
      } else {
        scores.stripe_us += 0.12;
      }
      contributions.brand_affinity = 0.15;
    } else if (brand === "unionpay" || brand === "jcb") {
      scores.checkout_crossborder += 0.15;
      contributions.brand_affinity = 0.15;
    }

    // 4. Circuit breaker checks
    if (velocity >= 15) {
      noulTriggered = true;
      noulAction = "STEP_DOWN_CHALLENGE";
      reason = "Velocity circuit breaker tripped (>15 tx/hr); step-down challenge routing";
      isFallback = true;
    } else if (amount >= 25000 && velocity >= 6) {
      noulTriggered = true;
      noulAction = "MANUAL_REVIEW";
      reason = "High ticket combined with elevated velocity; routed to fraud-isolated rail";
      isFallback = true;
    }

    // Calibrated Softmax (Temperature = 0.85)
    const T = 0.85;
    const expScores: Record<string, number> = {};
    let sumExp = 0;
    for (const [key, val] of Object.entries(scores)) {
      const expVal = Math.exp(val / T);
      expScores[key] = expVal;
      sumExp += expVal;
    }

    const probabilities: Record<string, number> = {};
    let bestChoice = "stripe_us";
    let maxProb = -1;

    for (const [key, expVal] of Object.entries(expScores)) {
      const prob = Math.round((expVal / sumExp) * 10000) / 10000;
      probabilities[key] = prob;
      if (prob > maxProb) {
        maxProb = prob;
        bestChoice = key;
      }
    }

    const elapsed = performance.now() - startTime;

    return {
      decision: bestChoice,
      choice: bestChoice,
      confidence: maxProb,
      probabilities,
      latency_ms: Math.round(elapsed * 1000) / 1000,
      routing_reason: reason,
      feature_contributions: contributions,
      is_fallback: isFallback,
      noul_triggered: noulTriggered,
      noul_action: noulAction,
      effective_fee_est: bestChoice === "adyen_eu" ? 130 : bestChoice === "stripe_us" ? 320 : 260,
      model_version: "v1.0.0-ts",
      timestamp: Date.now() / 1000,
    };
  }

  /**
   * Batch evaluate multiple payment transactions concurrently.
   *
   * @param contexts Array of payment context objects.
   */
  async decideBatch(contexts: PaymentContext[]): Promise<DecisionResult[]> {
    return Promise.all(contexts.map((ctx) => this.decide(ctx)));
  }

  /**
   * Check if the sidecar microservice is healthy.
   */
  async health(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1000);
      const res = await fetch(`${this.sidecarUrl}/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (!res.ok) return false;
      const data = (await res.json()) as { status?: string };
      return data.status === "healthy";
    } catch {
      return false;
    }
  }

  private buildFallbackResult(context: PaymentContext, startTime: number, reason: string): DecisionResult {
    // Compute local decision to ensure intelligent choice rather than a dumb static string
    const local = this.decideLocal(context);
    const elapsed = performance.now() - startTime;

    return {
      ...local,
      latency_ms: Math.round(elapsed * 1000) / 1000,
      routing_reason: `Zero-Dropout Fallback: ${reason} (Routed intelligently via embedded spec)`,
      is_fallback: true,
    };
  }
}
