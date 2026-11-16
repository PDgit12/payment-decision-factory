/**
 * Real-world Fintech Checkout Integration Example.
 *
 * Demonstrates pre-gateway routing evaluation in a production checkout handler.
 */

import { PaymentDecisionClient } from "../src/client.ts";
import type { PaymentContext } from "../src/types.ts";

async function processCheckout(cart: {
  amount: number;
  currency: string;
  cardBrand: string;
  binCountry: string;
  mcc: string;
  userId: string;
}) {
  console.log(`\n======================================================`);
  console.log(`🛒 Processing checkout for customer ${cart.userId}`);
  console.log(`💵 Amount: ${cart.amount} ${cart.currency} | Card: ${cart.cardBrand} (${cart.binCountry})`);

  // 1. Initialize Decision Client with 20ms SLA budget
  const decisionClient = new PaymentDecisionClient({
    sidecarUrl: "http://localhost:8080",
    timeoutMs: 20,
    enableLocalFallback: true, // guarantees <20ms zero-dropout routing
  });

  // 2. Prepare PaymentContext
  const context: PaymentContext = {
    amount: cart.amount,
    currency: cart.currency,
    card_brand: cart.cardBrand,
    bin_country: cart.binCountry,
    mcc: cart.mcc,
    velocity_1h: 1,
    customer_id: cart.userId,
  };

  // 3. Evaluate Routing Decision before Gateway Dispatch
  const t0 = performance.now();
  const decision = await decisionClient.decide(context);
  const evaluationTime = (performance.now() - t0).toFixed(3);

  console.log(`⚡ Decision Engine SLA: ${evaluationTime}ms (Engine reported: ${decision.latency_ms}ms)`);
  console.log(`🎯 Chosen Gateway Rail: [${decision.decision}]`);
  console.log(`📊 Calibrated Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
  console.log(`💡 Rationale: ${decision.routing_reason}`);
  console.log(`🛡️  Fallback Activated: ${decision.is_fallback}`);

  // 4. Dispatch to Chosen Gateway Provider
  switch (decision.decision) {
    case "stripe_us":
      console.log(`🚀 Dispatching charge to Stripe US API endpoint...`);
      return { status: "success", gateway: "stripe", tx_id: "ch_stripe_mock_123" };

    case "adyen_eu":
      console.log(`🚀 Dispatching charge to Adyen European Direct Clearing Rail (PSD2/SCA optimized)...`);
      return { status: "success", gateway: "adyen", tx_id: "adyen_psp_mock_456" };

    case "checkout_crossborder":
      console.log(`🚀 Dispatching charge to Checkout.com Global Multi-currency Rail...`);
      return { status: "success", gateway: "checkout_com", tx_id: "cko_pay_mock_789" };

    default:
      console.log(`⚠️ Dispatching to default backup gateway...`);
      return { status: "success", gateway: "stripe_default", tx_id: "backup_mock_000" };
  }
}

async function main() {
  // Test Case 1: Domestic US Payment
  await processCheckout({
    amount: 85.0,
    currency: "USD",
    cardBrand: "amex",
    binCountry: "US",
    mcc: "5411",
    userId: "usr_ny_01",
  });

  // Test Case 2: European Cross-Border Payment (PSD2/SCA)
  await processCheckout({
    amount: 140.0,
    currency: "EUR",
    cardBrand: "mastercard",
    binCountry: "DE",
    mcc: "5411",
    userId: "usr_berlin_02",
  });

  // Test Case 3: International APAC Cross-Border Payment
  await processCheckout({
    amount: 850.0,
    currency: "JPY",
    cardBrand: "jcb",
    binCountry: "JP",
    mcc: "5732",
    userId: "usr_tokyo_03",
  });
}

main().catch(console.error);
