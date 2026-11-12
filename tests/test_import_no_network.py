"""Importing tradingagents packages must never trigger network calls.

Each case runs in a fresh interpreter so previously imported modules in the
pytest process cannot mask import-time side effects. Inside the subprocess,
socket primitives are replaced with guards that record any connection
attempt before the package is imported.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

GUARD_SCRIPT = """
import socket

attempts = []

def _blocked(*args, **kwargs):
    attempts.append(args)
    raise RuntimeError("network access attempted during import")

socket.socket.connect = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

import {module}

print("ATTEMPTED_CONNECTIONS:", len(attempts))
print("IMPORT_OK")
"""


@pytest.mark.parametrize(
    "module",
    ["tradingagents.dataflows", "tradingagents.agents", "tradingagents"],
)
def test_import_does_not_touch_network(module):
    result = subprocess.run(
        [sys.executable, "-c", GUARD_SCRIPT.format(module=module)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"import {module} failed:\n{output}"
    assert "IMPORT_OK" in result.stdout, f"import {module} did not complete:\n{output}"
    assert "ATTEMPTED_CONNECTIONS: 0" in result.stdout, (
        f"import {module} attempted network connections:\n{output}"
    )
    assert "Error fetching" not in output, (
        f"import {module} triggered Alpaca API calls:\n{output}"
    )
