"""
Payment Decision Multi-Agent Orchestration Engine.

Exports DAG pipeline orchestrator, stages, transaction context accumulator,
circuit breakers, and payment agent base classes.
"""

from .agent_base import (
    CallableAgent,
    EngineAgentAdapter,
    MockAgent,
    PaymentAgent,
    SpecialistSDMAgentAdapter,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitMetrics,
    CircuitState,
)
from .context import (
    AgentExecutionResult,
    AuditEvent,
    ExecutionStatus,
    StageExecutionResult,
    TransactionContext,
)
from .pipeline import (
    DAGNode,
    DAGPipelineBuilder,
    PaymentOrchestrationPipeline,
    build_default_payment_pipeline,
    build_live_production_pipeline,
    build_specialist_sdm_pipeline,
)
from .stages import (
    ConditionalStage,
    ParallelStage,
    PipelineStage,
    SequentialStage,
)
from .closed_loop import (
    CanaryDeploymentAgent,
    ClosedLoopLifecycleResult,
    ClosedLoopOrchestrator,
    IngestionReport,
    LifecyclePhase,
    RolloutMode,
    SealedPackageMetadata,
    SelfAdjustmentDelta,
    SimulationReport,
)
from .gpu_factory_orchestrator import (
    ClusterScheduler,
    DeliveryPhase,
    DistributedGPUTrainingCoordinator,
    DistributedTrainingJobConfig,
    GPUEtlBenchmarkReport,
    GPUEtlPipeline,
    GPUInstanceCatalog,
    GPUInstanceSpec,
    GPUInstanceTier,
    GPU_INSTANCE_CATALOG,
    PrecisionMode,
    TrainingRunEstimate,
    Turnkey48HourPipelineScheduler,
    TurnkeyDeliveryPlan,
)
from .contractual_quotas import (
    AcquirerContractualQuota,
    ConstrainedQuotaRoutingOptimizer,
)

__all__ = [
    # Context
    "TransactionContext",
    "AgentExecutionResult",
    "StageExecutionResult",
    "AuditEvent",
    "ExecutionStatus",
    # Circuit Breakers
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitMetrics",
    "CircuitState",
    # Agents
    "PaymentAgent",
    "EngineAgentAdapter",
    "CallableAgent",
    "MockAgent",
    "SpecialistSDMAgentAdapter",
    # Stages
    "PipelineStage",
    "ParallelStage",
    "SequentialStage",
    "ConditionalStage",
    # Pipeline & DAG
    "PaymentOrchestrationPipeline",
    "DAGPipelineBuilder",
    "DAGNode",
    "build_default_payment_pipeline",
    "build_live_production_pipeline",
    "build_specialist_sdm_pipeline",
    # Closed-Loop Platform
    "ClosedLoopOrchestrator",
    "ClosedLoopLifecycleResult",
    "LifecyclePhase",
    "RolloutMode",
    "CanaryDeploymentAgent",
    "IngestionReport",
    "SimulationReport",
    "SelfAdjustmentDelta",
    "SealedPackageMetadata",
    # GPU Factory Orchestrator
    "GPUInstanceTier",
    "GPUInstanceSpec",
    "GPU_INSTANCE_CATALOG",
    "GPUInstanceCatalog",
    "ClusterScheduler",
    "PrecisionMode",
    "DeliveryPhase",
    "GPUEtlPipeline",
    "GPUEtlBenchmarkReport",
    "DistributedTrainingJobConfig",
    "TrainingRunEstimate",
    "DistributedGPUTrainingCoordinator",
    "Turnkey48HourPipelineScheduler",
    "TurnkeyDeliveryPlan",
]
