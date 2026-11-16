/**
 * Unit and integration tests for PaymentDecisionClient.
 */

import assert from "node:assert";
import test from "node:test";
import * as path from "node:path";
import * as fs from "node:fs";
import * as os from "node:os";
import * as crypto from "node:crypto";
import { PaymentDecisionClient } from "../src/client.ts";
import { SpecialistSDMClient } from "../src/sdm.ts";
import { SealedSDMPackageClient } from "../src/sealed.ts";
import type { PaymentContext } from "../src/types.ts";

test("PaymentDecisionClient - local synchronous evaluation", () => {
  const client = new PaymentDecisionClient({ localOnly: true });

  const usContext: PaymentContext = {
    amount: 50.0,
    currency: "USD",
    card_brand: "visa",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 1,
  };

  const result = client.decideLocal(usContext);
  assert.strictEqual(result.decision, "stripe_us");
  assert.ok(result.confidence > 0.5);
  assert.strictEqual(result.is_fallback, false);
  assert.ok(result.latency_ms < 10.0);
});

test("PaymentDecisionClient - European transaction routing", () => {
  const client = new PaymentDecisionClient({ localOnly: true });

  const euContext: PaymentContext = {
    amount: 120.0,
    currency: "EUR",
    card_brand: "mastercard",
    bin_country: "FR",
    mcc: "5411",
    velocity_1h: 1,
  };

  const result = client.decideLocal(euContext);
  assert.strictEqual(result.decision, "adyen_eu");
  assert.ok(result.confidence > 0.5);
});

test("PaymentDecisionClient - Cross-border transaction routing", () => {
  const client = new PaymentDecisionClient({ localOnly: true });

  const apacContext: PaymentContext = {
    amount: 300.0,
    currency: "AUD",
    card_brand: "unionpay",
    bin_country: "AU",
    mcc: "5732",
    velocity_1h: 1,
  };

  const result = client.decideLocal(apacContext);
  assert.strictEqual(result.decision, "checkout_crossborder");
  assert.ok(result.confidence > 0.5);
});

test("PaymentDecisionClient - Circuit breaker triggering on high velocity", () => {
  const client = new PaymentDecisionClient({ localOnly: true });

  const attackContext: PaymentContext = {
    amount: 25.0,
    currency: "USD",
    card_brand: "visa",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 20, // Exceeds threshold of 15
  };

  const result = client.decideLocal(attackContext);
  assert.strictEqual(result.is_fallback, true);
  assert.strictEqual(result.noul_triggered, true);
  assert.strictEqual(result.noul_action, "STEP_DOWN_CHALLENGE");
});

test("PaymentDecisionClient - Zero-dropout fallback when sidecar offline", async () => {
  const client = new PaymentDecisionClient({
    sidecarUrl: "http://127.0.0.1:59999", // non-existent port
    timeoutMs: 15,
    enableLocalFallback: true,
  });

  const context: PaymentContext = {
    amount: 75.0,
    currency: "USD",
    card_brand: "amex",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 1,
  };

  const result = await client.decide(context);
  assert.strictEqual(result.decision, "stripe_us");
  assert.strictEqual(result.is_fallback, true);
  assert.ok(result.routing_reason.includes("Fallback"));
});

test("SpecialistSDMClient - Routing Specialist evaluation (US)", () => {
  const client = new SpecialistSDMClient({ specialistType: "routing", inProcess: true });
  const ctx: PaymentContext = {
    amount: 80.0,
    currency: "USD",
    card_brand: "visa",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 1,
  };
  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "stripe_us");
  assert.ok(res.confidence > 0.5);
  assert.ok(res.latency_ms < 5.0);
});

test("SpecialistSDMClient - Routing Specialist evaluation (EU)", () => {
  const client = new SpecialistSDMClient({ specialistType: "routing", inProcess: true });
  const ctx: PaymentContext = {
    amount: 140.0,
    currency: "EUR",
    card_brand: "mastercard",
    bin_country: "DE",
    mcc: "5411",
    velocity_1h: 1,
  };
  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "adyen_eu");
  assert.ok(res.confidence > 0.5);
});

test("SpecialistSDMClient - Fraud Risk Specialist evaluation", () => {
  const client = new SpecialistSDMClient({ specialistType: "fraud", inProcess: true });
  const ctx: PaymentContext = {
    amount: 25.0,
    currency: "USD",
    card_brand: "visa",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 18, // Anomaly burst
  };
  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "high_risk_burst");
  assert.ok(res.confidence > 0.5);
});

test("SpecialistSDMClient - 3DS/SCA Exemption Specialist evaluation", () => {
  const client = new SpecialistSDMClient({ specialistType: "sca", inProcess: true });
  const ctx: PaymentContext = {
    amount: 15.0, // Low value < €30
    currency: "EUR",
    card_brand: "visa",
    bin_country: "FR",
    mcc: "5411",
    velocity_1h: 1,
  };
  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "frictionless_low_value");
  assert.ok(res.confidence > 0.5);
});

test("SpecialistSDMClient - Smart Retry Specialist evaluation", () => {
  const client = new SpecialistSDMClient({ specialistType: "retry", inProcess: true });
  const ctx: PaymentContext = {
    amount: 45.0,
    currency: "USD",
    card_brand: "visa",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 1,
  };
  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "immediate");
  assert.ok(res.confidence > 0.5);
});

function getTestPkgPath(): string {
  const candidates = [
    path.resolve("artifacts/test_sealed_package.sdmpkg"),
    path.resolve("../../artifacts/test_sealed_package.sdmpkg"),
    path.resolve("../artifacts/test_sealed_package.sdmpkg"),
    path.resolve("payment_decision_factory/artifacts/test_sealed_package.sdmpkg"),
    "/Users/piyushdua/payment_decision_factory/artifacts/test_sealed_package.sdmpkg",
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return path.resolve("payment_decision_factory/artifacts/test_sealed_package.sdmpkg");
}

test("SealedSDMPackageClient - in-memory decryption and evaluation", () => {
  const pkgPath = getTestPkgPath();
  const client = new SealedSDMPackageClient({
    packagePath: pkgPath,
    licenseKey: "test-ts-license-key-12345",
    clientId: "client_ts_corp",
  });

  assert.strictEqual(client.manifest.client_id, "client_ts_corp");
  assert.strictEqual(client.choices.length, 3);

  const ctx: PaymentContext = {
    amount: 99.0,
    currency: "USD",
    card_brand: "amex",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 1,
  };
  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "stripe_us");
  assert.ok(res.confidence > 0.5);
  assert.ok(res.latency_ms < 5.0);
});

test("SealedSDMPackageClient - invalid license key throws error", () => {
  const pkgPath = getTestPkgPath();
  assert.throws(() => {
    new SealedSDMPackageClient({
      packagePath: pkgPath,
      licenseKey: "wrong-invalid-key-xyz",
    });
  }, /authentication failed/);
});

test("SealedSDMPackageClient - client entitlement mismatch throws error", () => {
  const pkgPath = getTestPkgPath();
  assert.throws(() => {
    new SealedSDMPackageClient({
      packagePath: pkgPath,
      licenseKey: "test-ts-license-key-12345",
      clientId: "unauthorized_attacker_corp",
    });
  }, /entitlement mismatch/);
});

test("SealedSDMPackageClient - Enterprise Sub-Billion SDM tier sub-10ms evaluation", () => {
  const tmpDir = os.tmpdir();
  const pkgPath = path.join(tmpDir, `enterprise_test_${Date.now()}.sdmpkg`);

  const manifest = {
    format: "SDMPKG/1.0",
    package_id: "sdmpkg_enterprise_sub_billion_01",
    client_id: "enterprise_global_bank",
    domain: "enterprise_payment_routing",
    model_type: "enterprise_sub_billion_sdm",
    tier: "sub_billion",
    created_at: 1700000000,
    expires_at: 2000000000,
    sha256_checksum: "",
    input_dim: 128,
    num_classes: 3,
    choices: ["stripe_us", "adyen_eu", "checkout_crossborder"],
    spec: { tier: "sub_billion", architecture: "FT-Transformer", parameter_count: 120000000 },
  };

  const dummyModelBytes = Buffer.alloc(1024 * 512, 0x5a); // 512KB mock weights
  manifest.sha256_checksum = crypto.createHash("sha256").update(dummyModelBytes).digest("hex");

  const licenseKey = "ent-secret-license-key-777";
  const salt = crypto.randomBytes(16);
  const iv = crypto.randomBytes(12);
  const key = crypto.pbkdf2Sync(licenseKey, salt, 100000, 32, "sha256");

  function serializeManifestAAD(obj: any): string {
    if (Array.isArray(obj)) {
      return "[" + obj.map(serializeManifestAAD).join(", ") + "]";
    }
    if (obj !== null && typeof obj === "object") {
      const keys = Object.keys(obj).sort();
      return "{" + keys.map((k) => JSON.stringify(k) + ": " + serializeManifestAAD(obj[k])).join(", ") + "}";
    }
    return JSON.stringify(obj);
  }

  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
  cipher.setAAD(Buffer.from(serializeManifestAAD(manifest), "utf-8"));
  const ciphertext = Buffer.concat([cipher.update(dummyModelBytes), cipher.final()]);
  const tag = cipher.getAuthTag();

  fs.writeFileSync(
    pkgPath,
    JSON.stringify({
      format: "SDMPKG/1.0",
      manifest,
      crypto: {
        algorithm: "AES-256-GCM",
        kdf: "PBKDF2-HMAC-SHA256",
        kdf_iterations: 100000,
        salt_hex: salt.toString("hex"),
        iv_hex: iv.toString("hex"),
        tag_hex: tag.toString("hex"),
      },
      ciphertext: ciphertext.toString("base64"),
    })
  );

  const client = new SealedSDMPackageClient({
    packagePath: pkgPath,
    licenseKey,
    clientId: "enterprise_global_bank",
  });

  assert.strictEqual(client.packageId, "sdmpkg_enterprise_sub_billion_01");
  assert.strictEqual(client.tier, "sub_billion");
  assert.strictEqual(client.isEnterpriseTier, true);
  assert.strictEqual(client.domain, "enterprise_payment_routing");

  const ctx: PaymentContext = {
    amount: 1250.0,
    currency: "USD",
    card_brand: "visa",
    bin_country: "US",
    mcc: "5411",
    velocity_1h: 2,
  };

  const res = client.decideSync(ctx);
  assert.strictEqual(res.decision, "stripe_us");
  assert.ok(res.confidence > 0.5);
  assert.ok(res.latency_ms < 5.0, `Expected latency < 5.0ms, got ${res.latency_ms}ms`);
  assert.ok(res.routing_reason.includes("sub_billion"));

  // Batch evaluation
  for (let i = 0; i < 20; i++) {
    const bRes = client.decideSync(ctx);
    assert.ok(bRes.latency_ms < 10.0);
  }

  fs.unlinkSync(pkgPath);
});

