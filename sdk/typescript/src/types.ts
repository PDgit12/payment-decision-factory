/**
 * Payment Decision SDK - TypeScript Definitions.
 *
 * Strict typing for payment context inputs and calibrated decision outputs.
 */

export interface PaymentContext {
  /** Transaction amount in primary currency units (e.g., 49.99). */
  amount: number;
  /** ISO 4217 3-letter currency code (e.g., "USD", "EUR", "GBP"). */
  currency: string;
  /** Payment card network brand (e.g., "visa", "mastercard", "amex"). */
  card_brand: string;
  /** ISO 3166-1 alpha-2 card issuing country code (e.g., "US", "GB", "DE"). */
  bin_country: string;
  /** 4-digit ISO 18245 Merchant Category Code (e.g., "5411", "5732"). */
  mcc: string;
  /** Number of transaction attempts by user/fingerprint in last 60 minutes. */
  velocity_1h?: number;
  /** Optional customer or user identifier. */
  customer_id?: string;
  /** Optional tokenized payment method identifier. */
  payment_method_id?: string;
  /** Custom arbitrary key-value metadata. */
  metadata?: Record<string, unknown>;
}

export interface DecisionResult {
  /** Chosen gateway routing destination (e.g., "stripe_us", "adyen_eu", "checkout_crossborder"). */
  decision: string;
  /** Alias for decision property. */
  choice: string;
  /** Calibrated confidence probability (0.0 to 1.0). */
  confidence: number;
  /** Probability distribution across all candidate gateways. */
  probabilities: Record<string, number>;
  /** Execution latency in milliseconds. Guaranteed <20ms SLA. */
  latency_ms: number;
  /** Primary decision rationale or explainability rule. */
  routing_reason: string;
  /** Feature attribution impact scores. */
  feature_contributions?: Record<string, number>;
  /** True if decision was produced by safety fallback or circuit breaker. */
  is_fallback: boolean;
  /** True if 'noul' circuit breaker or step-down was triggered. */
  noul_triggered?: boolean;
  /** Action code if 'noul' rule was triggered. */
  noul_action?: string | null;
  /** Estimated effective gateway processing cost in basis points. */
  effective_fee_est?: number;
  /** Version of the deployed decision spec. */
  model_version: string;
  /** Unix timestamp in seconds. */
  timestamp: number;
}

export interface DecisionClientOptions {
  /** Sidecar endpoint URL. Defaults to "http://localhost:8080". */
  sidecarUrl?: string;
  /** Strict latency budget timeout in milliseconds. Defaults to 20ms. */
  timeoutMs?: number;
  /** Gateway choice to fallback to if sidecar times out or errors. Defaults to "stripe_us". */
  fallbackDecision?: string;
  /** Enable embedded in-process rule evaluator as zero-dropout fallback. Defaults to true. */
  enableLocalFallback?: boolean;
  /** Force local evaluation mode without making network requests. Defaults to false. */
  localOnly?: boolean;
}
