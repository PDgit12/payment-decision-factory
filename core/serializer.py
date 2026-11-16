"""
Payment Semantic State Serializer
==================================
Converts raw payment JSON and tabular transaction payloads into calibrated,
normalized semantic representations for System 1 fast-heuristic decision models.
Transforms raw values into continuous normalized vectors, categorized corridors,
and natural language contextual state representations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

from payment_decision_factory.core.spec import DecisionSpec


# Region mapping definitions
US_COUNTRIES = {"US", "CA", "MX"}
EU_COUNTRIES = {
    "GB", "DE", "FR", "NL", "IE", "ES", "IT", "SE", "NO", "DK",
    "BE", "AT", "CH", "PL", "PT", "FI", "GR", "CZ", "RO", "HU"
}
APAC_COUNTRIES = {"JP", "SG", "AU", "NZ", "HK", "KR", "TW", "MY", "TH"}
LATAM_COUNTRIES = {"BR", "AR", "CO", "CL", "PE"}

# Recognized vocabularies for one-hot vector representations
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "SEK", "OTHER"]
BRANDS = ["visa", "mastercard", "amex", "discover", "unionpay", "jcb", "other"]
REGIONS = ["US", "EU", "APAC", "LATAM", "OTHER"]
MCC_CATEGORIES = ["RETAIL_GROCERY", "DIGITAL_SAAS", "TRAVEL_AIRLINES", "FINANCIAL_SERVICES", "GAMBLING", "GENERAL"]


def get_country_region(country: str) -> str:
    """Map ISO country code to macro region."""
    code = (country or "").upper()
    if code in US_COUNTRIES:
        return "US"
    if code in EU_COUNTRIES:
        return "EU"
    if code in APAC_COUNTRIES:
        return "APAC"
    if code in LATAM_COUNTRIES:
        return "LATAM"
    return "OTHER"


def classify_mcc(mcc: str) -> str:
    """Classify 4-digit MCC code into macro merchant category."""
    m = str(mcc).strip()
    if m in ("5411", "5311", "5912", "5812", "5814"):
        return "RETAIL_GROCERY"
    if m in ("5734", "5818", "5968", "7372", "7379"):
        return "DIGITAL_SAAS"
    if (m.isdigit() and 3000 <= int(m) <= 3300) or m in ("4511", "4722", "7011"):
        return "TRAVEL_AIRLINES"
    if m in ("6012", "6051", "6211", "6540"):
        return "FINANCIAL_SERVICES"
    if m in ("7995", "7800", "7801", "7802"):
        return "GAMBLING"
    return "GENERAL"


def classify_corridor(bin_country: str, currency: str) -> str:
    """Derive transaction geopolitical settlement corridor."""
    region = get_country_region(bin_country)
    curr = (currency or "").upper()

    if region == "US" and curr in ("USD", "CAD"):
        return "DOMESTIC_US"
    if region == "EU" and curr in ("EUR", "GBP", "CHF", "SEK", "NOK"):
        return "INTRA_EU"
    if region == "APAC" and curr in ("JPY", "AUD", "SGD", "NZD"):
        return "INTRA_APAC"
    return "CROSS_BORDER"


def classify_amount_tier(amount: float) -> str:
    """Discretize continuous amount into financial ticket size tiers."""
    if amount < 15.0:
        return "MICRO"
    if amount < 150.0:
        return "SMALL"
    if amount < 1000.0:
        return "MEDIUM"
    if amount < 10000.0:
        return "HIGH"
    return "JUMBO"


def classify_velocity_tier(velocity_1h: int) -> str:
    """Categorize card velocity for risk assessment."""
    if velocity_1h <= 2:
        return "NORMAL"
    if velocity_1h <= 5:
        return "ELEVATED"
    if velocity_1h <= 14:
        return "HIGH"
    return "CRITICAL"


class SerializedPaymentState(BaseModel):
    """
    Calibrated semantic representation of a payment state.
    Designed for fast System 1 heuristic evaluation, explanation generation,
    and vector inference.
    """
    raw_input: Dict[str, Any]
    amount: float
    currency: str
    card_brand: str
    bin_country: str
    mcc: str
    velocity_1h: int

    # Semantic attributes
    corridor: str
    region: str
    amount_tier: str
    velocity_tier: str
    mcc_category: str
    risk_profile: str
    semantic_prompt: str

    # Normalized features and vector
    normalized_features: Dict[str, float]
    feature_vector: List[float] = Field(default_factory=list)
    feature_names: List[str] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a clean JSON-serializable dictionary."""
        return {
            "raw_input": self.raw_input,
            "amount": self.amount,
            "currency": self.currency,
            "card_brand": self.card_brand,
            "bin_country": self.bin_country,
            "mcc": self.mcc,
            "velocity_1h": self.velocity_1h,
            "corridor": self.corridor,
            "region": self.region,
            "amount_tier": self.amount_tier,
            "velocity_tier": self.velocity_tier,
            "mcc_category": self.mcc_category,
            "risk_profile": self.risk_profile,
            "semantic_prompt": self.semantic_prompt,
            "normalized_features": self.normalized_features,
            "feature_vector": self.feature_vector,
            "feature_names": self.feature_names,
        }

    @property
    def vector_array(self) -> np.ndarray:
        """Returns the feature vector as a numpy ndarray."""
        return np.array(self.feature_vector, dtype=np.float32)


class PaymentSerializer:
    """
    Serializer that transforms validated payment payloads into normalized
    semantic representations and vector embeddings for decision models.
    """

    def __init__(self, spec: Optional[DecisionSpec] = None):
        self.spec = spec
        self._build_feature_names()

    def _build_feature_names(self):
        """Construct deterministic feature vector column names."""
        names = ["amount_log", "amount_normalized", "velocity_normalized"]
        names.extend([f"curr_{c.lower()}" for c in CURRENCIES])
        names.extend([f"brand_{b.lower()}" for b in BRANDS])
        names.extend([f"region_{r.lower()}" for r in REGIONS])
        names.extend([f"mcc_{m.lower()}" for m in MCC_CATEGORIES])
        self.feature_names = names

    def serialize(self, payload: Dict[str, Any]) -> SerializedPaymentState:
        """
        Transform a raw or validated payment dictionary into a SerializedPaymentState.
        """
        # Validate through spec if available
        if self.spec:
            cleaned = self.spec.validate_input(payload)
        else:
            cleaned = payload

        amount = float(cleaned.get("amount", 0.0))
        currency = str(cleaned.get("currency", "USD")).upper()
        card_brand = str(cleaned.get("card_brand", "visa")).lower()
        bin_country = str(cleaned.get("bin_country", "US")).upper()
        mcc = str(cleaned.get("mcc", "5411"))
        velocity_1h = int(cleaned.get("velocity_1h", 0))

        # Semantic categorization
        region = get_country_region(bin_country)
        corridor = classify_corridor(bin_country, currency)
        amount_tier = classify_amount_tier(amount)
        velocity_tier = classify_velocity_tier(velocity_1h)
        mcc_category = classify_mcc(mcc)

        # Risk assessment profile
        if velocity_tier == "CRITICAL" or mcc_category in ("GAMBLING", "FINANCIAL_SERVICES") and amount > 5000:
            risk_profile = "CRITICAL"
        elif velocity_tier == "HIGH" or (amount_tier in ("HIGH", "JUMBO") and corridor == "CROSS_BORDER"):
            risk_profile = "ELEVATED"
        elif velocity_tier == "ELEVATED" or corridor == "CROSS_BORDER":
            risk_profile = "MODERATE"
        else:
            risk_profile = "LOW"

        # Continuous normalization
        # amount_log: log10(amount + 1) normalized up to $100,000 (log10 ~ 5)
        amount_log = math.log10(max(amount, 0.01) + 1.0)
        amount_norm = min(amount / 5000.0, 1.0)
        velocity_norm = min(velocity_1h / 20.0, 1.0)

        normalized_features: Dict[str, float] = {
            "amount_log": round(amount_log, 4),
            "amount_norm": round(amount_norm, 4),
            "velocity_norm": round(velocity_norm, 4),
            "is_domestic_us": 1.0 if corridor == "DOMESTIC_US" else 0.0,
            "is_intra_eu": 1.0 if corridor == "INTRA_EU" else 0.0,
            "is_cross_border": 1.0 if corridor == "CROSS_BORDER" else 0.0,
        }

        # Build one-hot vector representation
        vector = [round(amount_log, 4), round(amount_norm, 4), round(velocity_norm, 4)]

        # Currencies
        for c in CURRENCIES[:-1]:
            vector.append(1.0 if currency == c else 0.0)
        vector.append(1.0 if currency not in CURRENCIES[:-1] else 0.0)

        # Brands
        for b in BRANDS[:-1]:
            vector.append(1.0 if card_brand == b else 0.0)
        vector.append(1.0 if card_brand not in BRANDS[:-1] else 0.0)

        # Regions
        for r in REGIONS[:-1]:
            vector.append(1.0 if region == r else 0.0)
        vector.append(1.0 if region not in REGIONS[:-1] else 0.0)

        # MCC categories
        for m in MCC_CATEGORIES:
            vector.append(1.0 if mcc_category == m else 0.0)

        # Build natural language semantic prompt for System 1 contextual reasoning
        semantic_prompt = (
            f"[PAYMENT_STATE] amount: {amount:.2f} {currency} ({amount_tier}) | "
            f"card: {card_brand.upper()} issued in {bin_country} (Region: {region}) | "
            f"corridor: {corridor} | MCC: {mcc} ({mcc_category}) | "
            f"velocity_1h: {velocity_1h} ({velocity_tier}) | "
            f"risk_profile: {risk_profile}"
        )

        return SerializedPaymentState(
            raw_input=cleaned,
            amount=amount,
            currency=currency,
            card_brand=card_brand,
            bin_country=bin_country,
            mcc=mcc,
            velocity_1h=velocity_1h,
            corridor=corridor,
            region=region,
            amount_tier=amount_tier,
            velocity_tier=velocity_tier,
            mcc_category=mcc_category,
            risk_profile=risk_profile,
            semantic_prompt=semantic_prompt,
            normalized_features=normalized_features,
            feature_vector=vector,
            feature_names=self.feature_names,
        )

    def to_vector(self, payload: Dict[str, Any]) -> np.ndarray:
        """Directly convert payload to numpy feature vector."""
        state = self.serialize(payload)
        return state.vector_array

    def to_semantic_prompt(self, payload: Dict[str, Any]) -> str:
        """Directly convert payload to natural language semantic state prompt."""
        state = self.serialize(payload)
        return state.semantic_prompt

    def batch_serialize(self, payloads: List[Dict[str, Any]]) -> List[SerializedPaymentState]:
        """Serialize a batch of payment payloads."""
        return [self.serialize(p) for p in payloads]


_DEFAULT_SERIALIZER: Optional[PaymentSerializer] = None


def serialize_payment_record(payload: Dict[str, Any], spec: Optional[DecisionSpec] = None) -> np.ndarray:
    """Convenience helper to convert raw payment record to normalized feature vector."""
    global _DEFAULT_SERIALIZER
    if spec is not None:
        serializer = PaymentSerializer(spec)
    else:
        if _DEFAULT_SERIALIZER is None:
            _DEFAULT_SERIALIZER = PaymentSerializer()
        serializer = _DEFAULT_SERIALIZER
    return serializer.to_vector(payload)

