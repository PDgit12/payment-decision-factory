"""
Unit & Integration Tests for Multi-Tenant Isolated Factory & Private Marketplace
"""

import os
import sys
import pytest

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.orchestration.tenant_manager import TenantIsolationManager
from payment_decision_factory.core.compliance_guard import PCIDSSPANViolationError


def test_tenant_provisioning_and_isolation(tmp_path):
    mgr = TenantIsolationManager(base_tenants_dir=str(tmp_path / "tenants"))

    # Provision Tenant 1 (Acme EU)
    t1 = mgr.provision_tenant(tenant_id="acme_eu", company_name="Acme Europe SA", jurisdiction="EU")
    assert t1.tenant_id == "acme_eu"
    assert os.path.exists(t1.workspace_path)
    assert os.path.exists(os.path.join(t1.workspace_path, "vault"))
    assert os.path.exists(os.path.join(t1.workspace_path, "marketplace"))

    # Provision Tenant 2 (Nova US)
    t2 = mgr.provision_tenant(tenant_id="nova_us", company_name="Nova Payments LLC", jurisdiction="US")
    assert t2.tenant_id == "nova_us"

    # Verify Key Vault Isolation (Different root keys)
    v1 = mgr.get_tenant_vault("acme_eu")
    v2 = mgr.get_tenant_vault("nova_us")
    k1 = v1.list_keys()[0]
    k2 = v2.list_keys()[0]

    assert k1.client_id == "acme_eu"
    assert k2.client_id == "nova_us"
    assert k1.license_key != k2.license_key


def test_tenant_compliance_gate_blocks_dirty_data(tmp_path):
    mgr = TenantIsolationManager(base_tenants_dir=str(tmp_path / "tenants"))
    mgr.provision_tenant(tenant_id="bad_data_client", company_name="Bad Data Corp")

    dirty_csv = tmp_path / "dirty.csv"
    dirty_csv.write_text(
        "amount,currency,card_pan,target\n"
        "50.0,USD,4532015112830366,0\n",
        encoding="utf-8",
    )

    with pytest.raises(PCIDSSPANViolationError):
        mgr.execute_tenant_pipeline(
            tenant_id="bad_data_client",
            dataset_path=str(dirty_csv),
            pipeline_name="Illegal Pipeline",
        )


def test_tenant_clean_pipeline_execution(tmp_path):
    mgr = TenantIsolationManager(base_tenants_dir=str(tmp_path / "tenants"))
    mgr.provision_tenant(tenant_id="compliant_bank", company_name="Compliant Bank AG")

    sample_csv = os.path.join(_ROOT_DIR, "data", "creditcard_sample.csv")
    if not os.path.exists(sample_csv):
        pytest.skip("Sample credit card data not found")

    receipt = mgr.execute_tenant_pipeline(
        tenant_id="compliant_bank",
        dataset_path=sample_csv,
        pipeline_name="Smart Acquiring Gateway Router",
        archetype="smart_routing",
        parameter_tier="small",
        epochs=1,
    )

    assert receipt.status == "CERTIFIED_ENTERPRISE_READY"
    assert os.path.exists(receipt.sealed_package_path)

    # Verify model is registered strictly in Tenant's private marketplace
    v = mgr.get_tenant_vault("compliant_bank")
    models = v.list_models()
    assert len(models) >= 1
    assert models[0].client_id == "compliant_bank"
