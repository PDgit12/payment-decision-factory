"""
Unit Tests for PCI-DSS v4.0, GDPR & PSD2 Compliance Guard
"""

import os
import sys
import pytest

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from payment_decision_factory.core.compliance_guard import (
    ComplianceGuard,
    PCIDSSPANViolationError,
    GDPRPIIViolationError,
    luhn_checksum,
)


def test_luhn_checksum_algorithm():
    # Valid Visa card test number (standard test PAN)
    assert luhn_checksum("4532015112830366") is True
    # Invalid card number
    assert luhn_checksum("4532015112830367") is False
    # Non-card string
    assert luhn_checksum("1234") is False


def test_compliance_guard_rejects_raw_pan(tmp_path):
    dirty_csv = tmp_path / "dirty_pci_data.csv"
    dirty_csv.write_text(
        "transaction_id,amount,currency,card_pan,is_fraud\n"
        "tx_1,100.0,USD,4532015112830366,0\n",
        encoding="utf-8",
    )

    guard = ComplianceGuard(client_id="test_violator_client")

    with pytest.raises(PCIDSSPANViolationError) as exc_info:
        guard.scan_dataset_file(str(dirty_csv), enforce_fatal=True)

    assert "PCI-DSS" in str(exc_info.value)


def test_compliance_guard_rejects_unhashed_email(tmp_path):
    dirty_csv = tmp_path / "dirty_gdpr_data.csv"
    dirty_csv.write_text(
        "transaction_id,amount,currency,user_email,is_fraud\n"
        "tx_1,100.0,USD,john.doe@example.com,0\n",
        encoding="utf-8",
    )

    guard = ComplianceGuard(client_id="test_gdpr_client")

    with pytest.raises(GDPRPIIViolationError) as exc_info:
        guard.scan_dataset_file(str(dirty_csv), enforce_fatal=True)

    assert "GDPR PII Violation" in str(exc_info.value)


def test_compliance_guard_certifies_clean_dataset(tmp_path):
    clean_csv = tmp_path / "clean_tokenized_data.csv"
    clean_csv.write_text(
        "transaction_id,amount,currency,bin_country,token_card,mcc,is_fraud\n"
        "tx_1,100.0,USD,FR,tok_visa_8921,5411,0\n"
        "tx_2,250.0,EUR,DE,tok_mc_1142,5812,0\n"
        "tx_3,15.0,USD,US,tok_amex_4410,5999,1\n",
        encoding="utf-8",
    )

    guard = ComplianceGuard(client_id="clean_fintech_client")
    report = guard.scan_dataset_file(str(clean_csv), enforce_fatal=True)

    assert report.compliance_status == "COMPLIANT_CERTIFIED"
    assert report.pci_dss_passed is True
    assert report.gdpr_pii_passed is True
    assert len(report.violations) == 0
    assert report.sha256_audit_seal is not None
