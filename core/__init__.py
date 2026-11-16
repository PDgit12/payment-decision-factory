"""
Payment Decision Factory Core Engine & Primitives
==================================================
Exports:
- DecisionSpec: Declarative policy model and validator
- PaymentSerializer: Semantic state transformer
- DecisionEngine: Laya-style runtime (choice, score, noul)
- DecisionFactory: Pipeline runner and artifact compiler
- DecisionResult: Standardized decision object
"""

from payment_decision_factory.core.spec import (
    DecisionSpec,
    GatewayChoice,
    InputFieldSpec,
    SchemaSpec,
    SLASpec,
)
from payment_decision_factory.core.serializer import (
    PaymentSerializer,
    SerializedPaymentState,
)
from payment_decision_factory.core.engine import (
    DecisionEngine,
    DecisionResult,
    NoulResult,
    FeatureContribution,
    CalibrationMetrics,
)
from payment_decision_factory.core.factory import (
    DecisionFactory,
    CompiledArtifactBundle,
    ClientManifest,
    BenchmarkResult,
)
from payment_decision_factory.core.sdm import (
    SpecialistSDM,
    SpecialistPipeline,
    SpecialistTuningReport,
    export_specialist_to_onnx,
    ONNXSpecialistSDM,
)
from payment_decision_factory.core.sealed_package import (
    SealedPackage,
    DecryptedPackageContext,
    SealedModelRunner,
    LicenseAuthenticationError,
    LicenseExpiredError,
    SealedPackageSecurityError,
)
from payment_decision_factory.core.universal_factory import (
    UniversalDecisionFactory,
    UniversalDomainSpec,
    UniversalSDM,
    UniversalFeatureEncoder,
    UniversalTrainingReport,
)
from payment_decision_factory.core.enterprise_sdm import (
    EnterpriseSubBillionSDM,
    export_enterprise_sdm_to_onnx,
    EnterpriseONNXRunner,
)
from payment_decision_factory.core.gpu_enterprise_sdm import (
    EnterpriseGPUTabularTransformer,
    UVMCachedEntityEmbedding,
    FlashTabularAttentionBlock,
    GPUTrainingConfig,
    GPUTrainingPipeline,
    export_gpu_sdm_to_quantized_onnx,
)
from payment_decision_factory.core.premortem_defenses import (
    TwoSpeedLabelResolver,
    TwoSpeedResolvedRecord,
    HierarchicalEntityEmbedding,
    OnlineAcquirerDriftDetector,
    DriftStatus,
)

__all__ = [
    "DecisionSpec",
    "GatewayChoice",
    "InputFieldSpec",
    "SchemaSpec",
    "SLASpec",
    "PaymentSerializer",
    "SerializedPaymentState",
    "DecisionEngine",
    "DecisionResult",
    "NoulResult",
    "FeatureContribution",
    "CalibrationMetrics",
    "DecisionFactory",
    "CompiledArtifactBundle",
    "ClientManifest",
    "BenchmarkResult",
    "SpecialistSDM",
    "SpecialistPipeline",
    "SpecialistTuningReport",
    "export_specialist_to_onnx",
    "ONNXSpecialistSDM",
    "SealedPackage",
    "DecryptedPackageContext",
    "SealedModelRunner",
    "LicenseAuthenticationError",
    "LicenseExpiredError",
    "SealedPackageSecurityError",
    "UniversalDecisionFactory",
    "UniversalDomainSpec",
    "UniversalSDM",
    "UniversalFeatureEncoder",
    "UniversalTrainingReport",
    "EnterpriseSubBillionSDM",
    "export_enterprise_sdm_to_onnx",
    "EnterpriseONNXRunner",
    "EnterpriseGPUTabularTransformer",
    "UVMCachedEntityEmbedding",
    "FlashTabularAttentionBlock",
    "GPUTrainingConfig",
    "GPUTrainingPipeline",
    "export_gpu_sdm_to_quantized_onnx",
    "TwoSpeedLabelResolver",
    "TwoSpeedResolvedRecord",
    "HierarchicalEntityEmbedding",
    "OnlineAcquirerDriftDetector",
    "DriftStatus",
]
