"""
Payment Decision Specification Model
=====================================
Declarative schema and parsing engine for payment decision policies.
Validates inputs, choices, SLA limits, and Laya-style decision primitives.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


class InputFieldSpec(BaseModel):
    """Specification of an input feature field."""
    type: str = Field(..., description="Data type: 'float', 'integer', 'string', 'boolean'")
    required: bool = True
    default: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    enum: Optional[List[Union[str, int, float]]] = None
    regex: Optional[str] = None
    description: Optional[str] = None

    def validate_value(self, name: str, value: Any) -> Any:
        """Validate and cast a single input value against field specification."""
        if value is None:
            if self.required and self.default is None:
                raise ValueError(f"Missing required field '{name}'")
            return self.default

        # Type checking & casting
        if self.type in ("float", "number"):
            try:
                cast_val = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Field '{name}' must be a float, got {value!r}")
            if self.min is not None and cast_val < self.min:
                raise ValueError(f"Field '{name}' value {cast_val} below minimum {self.min}")
            if self.max is not None and cast_val > self.max:
                raise ValueError(f"Field '{name}' value {cast_val} above maximum {self.max}")
            return cast_val

        elif self.type in ("int", "integer"):
            try:
                cast_val = int(value)
            except (ValueError, TypeError):
                raise ValueError(f"Field '{name}' must be an integer, got {value!r}")
            if self.min is not None and cast_val < self.min:
                raise ValueError(f"Field '{name}' value {cast_val} below minimum {self.min}")
            if self.max is not None and cast_val > self.max:
                raise ValueError(f"Field '{name}' value {cast_val} above maximum {self.max}")
            return cast_val

        elif self.type in ("str", "string"):
            cast_val = str(value)
            if self.enum is not None and cast_val not in self.enum and cast_val.lower() not in [str(e).lower() for e in self.enum]:
                raise ValueError(f"Field '{name}' value {cast_val!r} not in allowed enum {self.enum}")
            if self.regex is not None and not re.match(self.regex, cast_val):
                raise ValueError(f"Field '{name}' value {cast_val!r} does not match regex pattern {self.regex}")
            return cast_val

        elif self.type in ("bool", "boolean"):
            if isinstance(value, bool):
                return value
            if str(value).lower() in ("true", "1", "yes"):
                return True
            if str(value).lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"Field '{name}' must be a boolean, got {value!r}")

        return value


class SchemaSpec(BaseModel):
    """Specification of all accepted input fields."""
    inputs: Dict[str, InputFieldSpec] = Field(default_factory=dict)


class SLASpec(BaseModel):
    """Latency and availability targets."""
    max_latency_ms: float = 25.0
    target_latency_ms: float = 12.0
    p99_latency_ms: float = 20.0
    timeout_action: str = "fallback"


class GatewayChoice(BaseModel):
    """Metadata and economics for a routing choice."""
    display_name: str
    provider: str
    region: str
    base_fee_bps: float = 290.0
    fixed_fee_cents: float = 30.0
    preferred_currencies: List[str] = Field(default_factory=list)
    preferred_countries: List[str] = Field(default_factory=list)
    max_single_tx: float = 500000.0
    sla_expected_ms: float = 15.0
    specialties: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChoicePrimitiveSpec(BaseModel):
    """Configuration for 'choice' routing primitive."""
    default_choice: str
    fallback_chain: Dict[str, List[str]] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "currency_affinity": 0.35,
            "geo_affinity": 0.40,
            "brand_affinity": 0.15,
            "cost_optimization": 0.10,
        }
    )
    affinities: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)


class ScorePrimitiveSpec(BaseModel):
    """Configuration for 'score' confidence calibration primitive."""
    scaling_method: str = "temperature_scaling"
    temperature: float = 0.85
    brier_score_target: float = 0.08
    confidence_floor: float = 0.40
    uncertainty_threshold: float = 0.38
    utility_factors: Dict[str, float] = Field(
        default_factory=lambda: {
            "acceptance_weight": 0.50,
            "cost_efficiency_weight": 0.30,
            "latency_weight": 0.20,
        }
    )


class NoulRule(BaseModel):
    """Single rule triggering circuit breaking or step-down."""
    name: str
    condition: str
    action: str
    reason: str


class NoulPrimitiveSpec(BaseModel):
    """Configuration for 'noul' no-action/circuit breaker primitive."""
    circuit_breaker_active: bool = True
    fallback_action: str = "STEP_DOWN_CHALLENGE"
    max_velocity_threshold: int = 15
    high_value_velocity_threshold: int = 6
    high_value_amount_threshold: float = 25000.0
    suspicious_mccs: List[str] = Field(default_factory=lambda: ["7995", "6051", "6211"])
    min_confidence_floor: float = 0.35
    rules: List[NoulRule] = Field(default_factory=list)


class PrimitivesSpec(BaseModel):
    """Collection of Laya-style decision primitives."""
    choice: ChoicePrimitiveSpec
    score: ScorePrimitiveSpec = Field(default_factory=ScorePrimitiveSpec)
    noul: NoulPrimitiveSpec = Field(default_factory=NoulPrimitiveSpec)


class CalibrationSpec(BaseModel):
    """Calibration targets for probability output."""
    method: str = "temperature_scaling"
    temperature: float = 0.85
    brier_score_target: float = 0.08
    expected_calibration_error: float = 0.035
    validation_sample_size: int = 1000


class DecisionSpec(BaseModel):
    """
    Complete declarative decision specification for Payment Decision Factory models.
    """
    spec_id: str
    name: str
    version: str = "1.0.0"
    domain: str = "payment_routing"
    description: Optional[str] = None
    sla: SLASpec = Field(default_factory=SLASpec)
    schema_spec: SchemaSpec = Field(..., alias="schema")
    choices: Dict[str, GatewayChoice] = Field(default_factory=dict)
    primitives: PrimitivesSpec
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    @property
    def schema(self) -> SchemaSpec:
        """Accessor for input schema specification."""
        return self.schema_spec

    @field_validator("choices")
    @classmethod
    def validate_choices_non_empty(cls, v: Dict[str, GatewayChoice]) -> Dict[str, GatewayChoice]:
        if not v:
            raise ValueError("DecisionSpec must define at least one choice.")
        return v

    @model_validator(mode="after")
    def validate_default_choice_in_choices(self) -> DecisionSpec:
        default_choice = self.primitives.choice.default_choice
        if default_choice not in self.choices:
            raise ValueError(
                f"Default choice '{default_choice}' must be defined in choices: {list(self.choices.keys())}"
            )
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecisionSpec:
        """Create a DecisionSpec directly from a dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path_or_stream: Union[str, Any]) -> DecisionSpec:
        """Load and validate a DecisionSpec from a YAML file path or YAML string content."""
        if isinstance(path_or_stream, str):
            try:
                with open(path_or_stream, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except (FileNotFoundError, OSError):
                # Try parsing as raw YAML string
                data = yaml.safe_load(path_or_stream)
        else:
            data = yaml.safe_load(path_or_stream)

        if not isinstance(data, dict):
            raise ValueError("YAML document must resolve to a dictionary.")
        return cls(**data)

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Export specification to YAML string or write to file."""
        data = self.model_dump(by_alias=True)
        yaml_content = yaml.dump(data, sort_keys=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
        return yaml_content

    def validate_input(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate incoming transaction payload against the spec's input schema.
        Returns normalized and type-cast payload dictionary.
        """
        validated = {}
        for field_name, field_spec in self.schema.inputs.items():
            raw_value = payload.get(field_name, field_spec.default)
            validated[field_name] = field_spec.validate_value(field_name, raw_value)

        # Retain extra metadata if present
        for k, v in payload.items():
            if k not in validated:
                validated[k] = v

        return validated
