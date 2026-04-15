"""Runtime path utilities for MiniVecDB.

This module centralises where MiniVecDB writes runtime artifacts:
- VectorStore run directories under project-root/db_run/
- Active run marker used by default CLI and VectorStore behavior
- Shared Hugging Face cache under project-root/db_run/model_cache/huggingface

Keeping these rules in one place avoids path drift across modules.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Optional

DB_RUN_DIRNAME = "db_run"
ACTIVE_RUN_FILENAME = ".active_run"
SHARED_DB_FILENAME = "minivecdb.db"
MODEL_CACHE_DIRNAME = "model_cache"
MODEL_CACHE_SUBDIR = os.path.join(MODEL_CACHE_DIRNAME, "huggingface")
DEFAULT_RUN_PREFIX = "demo"
PROJECT_ROOT_ENV = "MINIVECDB_PROJECT_ROOT"


def _sanitize_prefix(prefix: str) -> str:
    """Return a filesystem-safe run-name prefix."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix.strip())
    return safe or DEFAULT_RUN_PREFIX


def get_project_root() -> str:
    """Return the project root directory path.

    Uses MINIVECDB_PROJECT_ROOT when set, otherwise resolves from this file.
    """
    env_root = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    core_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(core_dir)


def get_db_run_root(project_root: Optional[str] = None) -> str:
    """Return the absolute path to the db_run root directory."""
    root = project_root or get_project_root()
    return os.path.join(root, DB_RUN_DIRNAME)


def ensure_db_run_root(project_root: Optional[str] = None) -> str:
    """Create db_run root if missing and return its absolute path."""
    db_run_root = get_db_run_root(project_root=project_root)
    os.makedirs(db_run_root, exist_ok=True)
    return db_run_root


def get_active_run_marker_path(project_root: Optional[str] = None) -> str:
    """Return path to the active-run marker file."""
    return os.path.join(
        ensure_db_run_root(project_root=project_root),
        ACTIVE_RUN_FILENAME,
    )


def is_within_db_run(path: str, project_root: Optional[str] = None) -> bool:
    """Return True when path is inside the db_run root directory."""
    db_run_root = os.path.abspath(ensure_db_run_root(project_root=project_root))
    candidate = os.path.abspath(path)
    return os.path.commonpath([candidate, db_run_root]) == db_run_root


def read_active_run_path(project_root: Optional[str] = None) -> Optional[str]:
    """Read and validate the active run path from marker file.

    Returns:
        Absolute path to active run directory when valid, otherwise None.
    """
    marker_path = get_active_run_marker_path(project_root=project_root)
    if not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as marker_file:
            raw_value = marker_file.read().strip()
    except OSError:
        return None

    if not raw_value:
        return None

    db_run_root = ensure_db_run_root(project_root=project_root)
    if os.path.isabs(raw_value):
        candidate = raw_value
    else:
        candidate = os.path.join(db_run_root, raw_value)

    candidate = os.path.abspath(candidate)

    # Reject special cache directory or anything outside db_run.
    if os.path.basename(candidate) == MODEL_CACHE_DIRNAME:
        return None
    if not is_within_db_run(candidate, project_root=project_root):
        return None
    if not os.path.isdir(candidate):
        return None

    return candidate


def write_active_run_name(run_name: str, project_root: Optional[str] = None) -> None:
    """Write a run directory name to the active-run marker file."""
    if not run_name or any(sep in run_name for sep in ("/", "\\")):
        raise ValueError("run_name must be a simple directory name.")

    marker_path = get_active_run_marker_path(project_root=project_root)
    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write(f"{run_name}\n")


def set_active_run_path(run_path: str, project_root: Optional[str] = None) -> None:
    """Mark a run directory as active.

    The path must exist and be located under db_run.
    """
    if not run_path:
        raise ValueError("run_path must be a non-empty string.")

    abs_path = os.path.abspath(os.path.expanduser(run_path))
    if not os.path.isdir(abs_path):
        raise ValueError("run_path must point to an existing directory.")
    if not is_within_db_run(abs_path, project_root=project_root):
        raise ValueError("run_path must be inside the db_run directory.")

    run_name = os.path.basename(abs_path)
    write_active_run_name(run_name, project_root=project_root)


def generate_run_name(prefix: str = DEFAULT_RUN_PREFIX) -> str:
    """Generate a unique run directory name.

    Format: <prefix>_<unix_timestamp>_<short_random>
    Example: demo_1765667200_a1b2c3
    """
    safe_prefix = _sanitize_prefix(prefix)
    timestamp = int(time.time())
    rand = uuid.uuid4().hex[:6]
    return f"{safe_prefix}_{timestamp}_{rand}"


def create_new_run_path(
    prefix: str = DEFAULT_RUN_PREFIX,
    project_root: Optional[str] = None,
) -> str:
    """Create a new unique run directory under db_run and mark it active."""
    db_run_root = ensure_db_run_root(project_root=project_root)

    for _ in range(50):
        run_name = generate_run_name(prefix=prefix)
        run_path = os.path.join(db_run_root, run_name)
        try:
            os.makedirs(run_path, exist_ok=False)
        except FileExistsError:
            continue

        write_active_run_name(run_name, project_root=project_root)
        return os.path.abspath(run_path)

    raise RuntimeError("Failed to create a unique run directory in db_run.")


def resolve_storage_path(
    storage_path: Optional[str],
    create_new_run: bool = False,
    run_prefix: str = DEFAULT_RUN_PREFIX,
    project_root: Optional[str] = None,
) -> str:
    """Resolve final storage path from explicit or managed runtime settings."""
    if storage_path is not None:
        return os.path.abspath(os.path.expanduser(storage_path))

    if create_new_run:
        return create_new_run_path(prefix=run_prefix, project_root=project_root)

    active_run = read_active_run_path(project_root=project_root)
    if active_run is not None:
        return active_run

    return create_new_run_path(prefix=run_prefix, project_root=project_root)


def get_model_cache_path(project_root: Optional[str] = None) -> str:
    """Return (and create) the shared Hugging Face cache path in project root."""
    db_run_root = ensure_db_run_root(project_root=project_root)
    cache_path = os.path.join(db_run_root, MODEL_CACHE_SUBDIR)
    os.makedirs(cache_path, exist_ok=True)
    return os.path.abspath(cache_path)


def get_shared_db_path(project_root: Optional[str] = None) -> str:
    """Return the absolute path to the shared SQLite database (db_run/minivecdb.db)."""
    db_run_root = ensure_db_run_root(project_root=project_root)
    return os.path.abspath(os.path.join(db_run_root, SHARED_DB_FILENAME))


def list_run_directories(project_root: Optional[str] = None) -> list:
    """Return absolute paths of all run directories under db_run, sorted by mtime desc.

    Skips the model cache directory and any hidden files. Returns [] when db_run
    does not exist or holds no run directories yet.
    """
    db_run_root = get_db_run_root(project_root=project_root)
    if not os.path.isdir(db_run_root):
        return []

    entries = []
    for name in os.listdir(db_run_root):
        if name.startswith(".") or name == MODEL_CACHE_DIRNAME:
            continue
        full = os.path.join(db_run_root, name)
        if not os.path.isdir(full):
            continue
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        entries.append((mtime, os.path.abspath(full)))

    entries.sort(key=lambda t: t[0], reverse=True)
    return [path for _, path in entries]
