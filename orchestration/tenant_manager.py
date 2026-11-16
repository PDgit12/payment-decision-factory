"""
Multi-Tenant Isolated Factory & Personalized Marketplace Orchestrator
=====================================================================
Ensures complete physical and logical isolation between enterprise clients:
1. Isolated Tenant Sandboxes: Dedicated directory trees, configs, and storage per client.
2. Compliance-First Gatekeeper: Runs PCI-DSS & GDPR validation prior to compute allocation.
3. Private Tenant Marketplaces: Client A's marketplace contains only Client A's models.
4. Zero-Leakage Cryptographic Key Vaults: Keys are cryptographically bound to tenant IDs.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Unshadow standard library code module if local code.py exists
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code):
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_FACTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [_ROOT_DIR, _FACTORY_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from payment_decision_factory.core.compliance_guard import (
    ComplianceAuditReport,
    ComplianceGuard,
    ComplianceViolationError,
)
from payment_decision_factory.studio.vault import (
    ClientLicenseKey,
    MarketplaceModelRecord,
    ModelMarketplaceVault,
)
from payment_decision_factory.universal_client_pipeline import (
    ClientPipelineIntent,
    UniversalClientPipelineEngine,
    UniversalTurnkeyReceipt,
)

logger = logging.getLogger("payment_decision_factory.tenant_manager")


@dataclass
class TenantProfile:
    """Metadata and configuration for a registered enterprise tenant."""

    tenant_id: str
    company_name: str
    jurisdiction: str  # e.g., 'EU', 'US', 'GLOBAL'
    created_at: str
    compliance_certified: bool
    workspace_path: str
    active_keys_count: int = 0
    models_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TenantIsolationManager:
    """
    Manages client onboarding, workspace isolation, compliance gates,
    and private model marketplaces.
    """

    def __init__(self, base_tenants_dir: Optional[str] = None):
        if base_tenants_dir is None:
            base_tenants_dir = os.path.join(_FACTORY_DIR, "artifacts", "tenants")
        self.base_dir = Path(base_tenants_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "tenants_registry.json"
        self._init_registry()

    def _init_registry(self):
        if not self.index_file.exists():
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_registry(self, data: Dict[str, Dict[str, Any]]):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def provision_tenant(
        self,
        tenant_id: str,
        company_name: str,
        jurisdiction: str = "GLOBAL",
    ) -> TenantProfile:
        """
        Creates an isolated tenant environment with dedicated directories:
        - compliance/ (audit trails and certs)
        - marketplace/ (their private .sdmpkg packages)
        - vault/ (their cryptographic keys)
        - data/ (sanitized input datasets)
        """
        tenant_ws = self.base_dir / tenant_id
        for subdir in ["compliance", "marketplace", "vault", "data", "models"]:
            (tenant_ws / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize tenant's private key vault with tenant identity
        tenant_vault = ModelMarketplaceVault(vault_dir=str(tenant_ws / "vault"), default_client_id=tenant_id)

        profile = TenantProfile(
            tenant_id=tenant_id,
            company_name=company_name,
            jurisdiction=jurisdiction,
            created_at=datetime.now(timezone.utc).isoformat(),
            compliance_certified=False,
            workspace_path=str(tenant_ws),
            active_keys_count=len(tenant_vault.list_keys()),
            models_count=0,
        )

        reg = self._load_registry()
        reg[tenant_id] = profile.to_dict()
        self._save_registry(reg)

        logger.info(f"Tenant provisioned successfully: {tenant_id} at {tenant_ws}")
        return profile

    def get_tenant_profile(self, tenant_id: str) -> Optional[TenantProfile]:
        reg = self._load_registry()
        d = reg.get(tenant_id)
        if d:
            return TenantProfile(**d)
        return None

    def list_tenants(self) -> List[TenantProfile]:
        reg = self._load_registry()
        return [TenantProfile(**d) for d in reg.values()]

    def get_tenant_vault(self, tenant_id: str) -> ModelMarketplaceVault:
        tenant_ws = self.base_dir / tenant_id
        if not tenant_ws.exists():
            raise FileNotFoundError(f"Tenant workspace not found for: {tenant_id}")
        return ModelMarketplaceVault(vault_dir=str(tenant_ws / "vault"), default_client_id=tenant_id)

    def run_compliance_audit(
        self,
        tenant_id: str,
        dataset_path: str,
        enforce_fatal: bool = True,
    ) -> ComplianceAuditReport:
        """
        Mandatory Compliance Gate: Must pass before training can proceed.
        """
        profile = self.get_tenant_profile(tenant_id)
        jurisdiction = profile.jurisdiction if profile else "GLOBAL"

        guard = ComplianceGuard(client_id=tenant_id, jurisdiction=jurisdiction)
        report = guard.scan_dataset_file(csv_path=dataset_path, enforce_fatal=enforce_fatal)

        # Save report in tenant's compliance directory
        tenant_ws = self.base_dir / tenant_id
        cert_path = tenant_ws / "compliance" / f"{report.audit_id}.json"
        with open(cert_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())

        # Update tenant profile compliance status
        if profile and report.compliance_status == "COMPLIANT_CERTIFIED":
            profile.compliance_certified = True
            reg = self._load_registry()
            reg[tenant_id] = profile.to_dict()
            self._save_registry(reg)

        return report

    def execute_tenant_pipeline(
        self,
        tenant_id: str,
        dataset_path: str,
        pipeline_name: str,
        archetype: str = "smart_routing",
        parameter_tier: str = "small",
        compute_provider: str = "mac_mps_local",
        epochs: int = 2,
    ) -> UniversalTurnkeyReceipt:
        """
        Compliance-First Pipeline Execution:
        1. Verifies compliance gate.
        2. Executes Universal Pipeline in tenant's isolated sandbox.
        3. Registers output in tenant's private marketplace.
        """
        tenant_ws = self.base_dir / tenant_id
        if not tenant_ws.exists():
            raise FileNotFoundError(f"Tenant workspace not found for: {tenant_id}")

        # Step 1: Mandatory Compliance Scan
        logger.info(f"Running pre-training compliance gate for tenant: {tenant_id}")
        audit_report = self.run_compliance_audit(tenant_id=tenant_id, dataset_path=dataset_path, enforce_fatal=True)

        if audit_report.compliance_status != "COMPLIANT_CERTIFIED":
            raise ComplianceViolationError(f"Compliance Gate Rejected dataset: {audit_report.violations}")

        # Step 2: Fetch tenant's active license key
        vault = self.get_tenant_vault(tenant_id)
        keys = vault.list_keys(client_id=tenant_id)
        active_key = keys[0].license_key if keys else f"lic_{tenant_id}_root"

        # Step 3: Run Engine in Tenant's Isolated Workspace
        intent = ClientPipelineIntent(
            client_id=tenant_id,
            pipeline_name=pipeline_name,
            archetype=archetype,
            parameter_tier=parameter_tier,
            compute_provider=compute_provider,
            license_key=active_key,
            output_dir=str(tenant_ws / "models"),
        )

        engine = UniversalClientPipelineEngine(intent=intent)
        receipt = engine.run(dataset_path=dataset_path, epochs=epochs)

        # Step 4: Register model strictly in tenant's private marketplace
        record = MarketplaceModelRecord(
            model_id=f"{tenant_id}_{archetype}_{parameter_tier}",
            client_id=tenant_id,
            model_name=f"{pipeline_name} ({parameter_tier.upper()})",
            parameter_tier=parameter_tier,
            parameter_count=801411 if parameter_tier == "small" else 5248000,
            raw_size_mb=round(os.path.getsize(receipt.sealed_package_path) / (1024.0**2), 2),
            package_path=receipt.sealed_package_path,
            sha256_audit_hash=receipt.sha256_audit_hash,
            status=receipt.status,
            validation_accuracy=receipt.validation_accuracy,
            brier_score=receipt.brier_score,
            p50_latency_ms=receipt.p50_latency_ms,
            p99_latency_ms=receipt.p99_latency_ms,
            created_at=receipt.timestamp,
            tags=[jurisdiction_tag for jurisdiction_tag in [archetype, parameter_tier, "pci_compliant"]],
        )
        vault.register_model(record)

        # Update registry counts
        profile = self.get_tenant_profile(tenant_id)
        if profile:
            profile.models_count = len(vault.list_models())
            reg = self._load_registry()
            reg[tenant_id] = profile.to_dict()
            self._save_registry(reg)

        return receipt
