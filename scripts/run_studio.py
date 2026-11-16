#!/usr/bin/env python3
"""
Payment Decision Studio - 1-Command Server Launcher
===================================================
Launches the interactive Cursor-like Decision Studio, Model Training Cockpit,
Personalized Marketplace, and Cryptographic Key Vault.

Usage:
    python scripts/run_studio.py
    python scripts/run_studio.py --port 8080 --open
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import webbrowser

# Robust unshadowing of standard library code module if local code.py exists
_stdlib_code = os.path.join(os.path.dirname(os.__file__), "code.py")
if os.path.exists(_stdlib_code):
    _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
    if _spec and _spec.loader:
        _m = importlib.util.module_from_spec(_spec)
        sys.modules["code"] = _m
        _spec.loader.exec_module(_m)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FACTORY_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_FACTORY_DIR, ".."))

for p in [_REPO_ROOT, _FACTORY_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Payment Decision Studio Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--open", action="store_true", help="Automatically open browser on launch")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}       PAYMENT DECISION STUDIO & PERSONALIZED MARKETPLACE             {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  {BOLD}Studio URL        :{RESET} {CYAN}{url}{RESET}")
    print(f"  {BOLD}Cursor Playground :{RESET} {url}/#cursor")
    print(f"  {BOLD}Training Cockpit  :{RESET} {url}/#cockpit")
    print(f"  {BOLD}Model Marketplace :{RESET} {url}/#marketplace")
    print(f"  {BOLD}Key Vault         :{RESET} {url}/#vault")
    print(f"  {BOLD}Hardware Engine   :{RESET} Apple Silicon MPS / Local GPU ($0 Cloud Cost)")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(
        "payment_decision_factory.studio.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
