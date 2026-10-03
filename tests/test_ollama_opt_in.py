"""The normal test command must never start real local inference."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OLLAMA_TEST = "tests/integration/test_ollama_backend.py"


def test_plain_pytest_deselects_real_ollama_test_during_collection() -> None:
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/test_models.py",
            OLLAMA_TEST,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert "1 deselected" in collected.stdout
    assert "test_local_qwen3_returns_a_constrained_decision" not in collected.stdout
