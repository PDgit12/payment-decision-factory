"""
Payment Decision Factory - Specialized Domain Agents.
=====================================================
Declarative domain decision agents for end-to-end payment lifecycle:
- `FraudScreenerAgent`: Pre-auth velocity, risk scoring, and card testing mitigation.
- `Dynamic3DSAgent`: PSD2 RTS SCA exemptions (TRA Article 18, Low-Value Article 16).
- `SmartRetryAgent`: Decline recovery, backoff window prediction, and scheme compliance.
- `SmartRoutingAgent`: Multi-rail payment gateway routing (Stripe, Adyen, Checkout.com).
"""

from .base import AgentDecisionResult, BaseDecisionAgent
from .dynamic_3ds import Dynamic3DSAgent
from .fraud_screener import FraudScreenerAgent
from .smart_retry import SmartRetryAgent
from .smart_routing import SmartRoutingAgent
from .universal import UniversalDecisionAgent

__all__ = [
    "BaseDecisionAgent",
    "AgentDecisionResult",
    "FraudScreenerAgent",
    "Dynamic3DSAgent",
    "SmartRetryAgent",
    "SmartRoutingAgent",
    "UniversalDecisionAgent",
    "create_agent",
]


def create_agent(agent_type: str, **kwargs) -> BaseDecisionAgent:
    """
    Factory helper to instantiate payment decision agents by key.

    Supported agent types:
    - 'fraud', 'fraud_screener': FraudScreenerAgent
    - '3ds', 'dynamic_3ds', 'sca': Dynamic3DSAgent
    - 'retry', 'smart_retry': SmartRetryAgent
    - 'routing', 'smart_routing': SmartRoutingAgent
    - 'universal', 'generic': UniversalDecisionAgent
    """
    t = agent_type.lower().strip()
    if t in ("fraud", "fraud_screener"):
        return FraudScreenerAgent(**kwargs)
    elif t in ("3ds", "dynamic_3ds", "sca"):
        return Dynamic3DSAgent(**kwargs)
    elif t in ("retry", "smart_retry", "recovery"):
        return SmartRetryAgent(**kwargs)
    elif t in ("routing", "smart_routing", "gateway"):
        return SmartRoutingAgent(**kwargs)
    elif t in ("universal", "generic", "general"):
        return UniversalDecisionAgent(**kwargs)
    else:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. Supported: fraud_screener, dynamic_3ds, smart_retry, smart_routing, universal"
        )
