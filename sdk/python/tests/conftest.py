"""Pytest configuration for payment_decision_sdk test suite."""

import sys
import os
import importlib.util

# Ensure standard library 'code' module is bound to stdlib rather than any local code.py
stdlib_dir = os.path.dirname(os.__file__)
stdlib_code_path = os.path.join(stdlib_dir, "code.py")
if os.path.exists(stdlib_code_path):
    spec = importlib.util.spec_from_file_location("code", stdlib_code_path)
    if spec and spec.loader:
        code_mod = importlib.util.module_from_spec(spec)
        sys.modules["code"] = code_mod
        spec.loader.exec_module(code_mod)

sdk_python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [sdk_python_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)
