"""Master test runner for MiniVecDB.

This runner executes test files in a fixed order and reports a summary of
passed/failed/skipped files. It supports two execution modes automatically:

- Without pytest installed:
  Runs standalone assert-based scripts and skips pytest-dependent files.
- With pytest installed:
  Runs pytest-dependent files via `python -m pytest <file> -v`.

You can also run the full pytest suite directly with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class FileRunResult:
    """Store execution details for a single test file."""

    file_name: str
    status: str
    return_code: int
    command: str


# Ordered execution list requested by the project.
TEST_FILES_IN_ORDER: List[str] = [
    "run_tests_distance.py",
    "run_tests_embeddings.py",
    "day2_3_integration_test.py",
    "test_distance_metrics.py",
    "test_day5.py",
    "test_day6.py",
    "test_day7.py",
    "test_day8.py",
    "test_file_processor.py",
    "test_file_processor_search_integration.py",
    "test_integration.py",
    "test_edge_cases.py",
]

# Files that need pytest to run correctly.
PYTEST_REQUIRED_FILES = {
    "test_distance_metrics.py",
    "test_day5.py",
    "test_day6.py",
    "test_day7.py",
    "test_day8.py",
    "test_file_processor.py",
    "test_file_processor_search_integration.py",
}


def _has_pytest() -> bool:
    """Return True if pytest is importable in the current environment."""
    return importlib.util.find_spec("pytest") is not None


def _run_subprocess(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and capture combined stdout/stderr for reporting."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _build_command(file_name: str, pytest_available: bool) -> List[str]:
    """Build the execution command for one test file."""
    if file_name in PYTEST_REQUIRED_FILES:
        if not pytest_available:
            return []
        return [sys.executable, "-m", "pytest", f"tests/{file_name}", "-v"]

    return [sys.executable, f"tests/{file_name}"]


def _run_test_file(file_name: str, project_root: Path, pytest_available: bool) -> FileRunResult:
    """Execute one test file and classify it as passed/failed/skipped."""
    command = _build_command(file_name, pytest_available)
    if not command:
        return FileRunResult(
            file_name=file_name,
            status="skipped",
            return_code=0,
            command="pytest not available",
        )

    completed = _run_subprocess(command, cwd=project_root)
    status = "passed" if completed.returncode == 0 else "failed"

    print("-" * 72)
    print(f"Running: {file_name}")
    print(f"Command: {' '.join(command)}")
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())

    return FileRunResult(
        file_name=file_name,
        status=status,
        return_code=completed.returncode,
        command=" ".join(command),
    )


def run_all_tests() -> int:
    """Run all test files in order and return process exit code."""
    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent
    pytest_available = _has_pytest()

    print("=" * 72)
    print("MiniVecDB Master Test Runner")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Pytest available: {pytest_available}")
    print()

    results: List[FileRunResult] = []
    for file_name in TEST_FILES_IN_ORDER:
        result = _run_test_file(file_name, project_root, pytest_available)
        results.append(result)
        if result.status == "skipped":
            print("-" * 72)
            print(f"Skipping: {file_name} (requires pytest)")

    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")

    print("=" * 72)
    print("MASTER SUMMARY")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total:   {len(results)}")
    print("=" * 72)

    if pytest_available:
        print("Tip: Full pytest run is also available with:")
        print(f"  {sys.executable} -m pytest tests/ -v")
    else:
        print("Tip: Install pytest to execute pytest-based files too:")
        print("  pip install pytest")

    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result.status == "failed":
                print(f"  - {result.file_name} (exit code {result.return_code})")
        return 1

    return 0


if __name__ == "__main__":
    """CLI entry point for the master runner."""
    raise SystemExit(run_all_tests())
