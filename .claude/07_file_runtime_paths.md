# MiniVecDB — File: `core/runtime_paths.py`

> **Location**: `minivecdb/core/runtime_paths.py`
> **Lines**: 233 | **Size**: 8 KB
> **Role**: Centralised path resolution for runtime artifacts, run management, and shared DB routing

---

## Purpose

This module is the **single source of truth** for where MiniVecDB writes files at runtime. It controls:
- `db_run/` root directory location
- Active run marker (`.active_run` file)
- Unique run folder creation with collision avoidance
- Shared DB path (`db_run/minivecdb.db`)
- Hugging Face model cache path
- Run directory enumeration

---

## Constants

```python
DB_RUN_DIRNAME = "db_run"
ACTIVE_RUN_FILENAME = ".active_run"
SHARED_DB_FILENAME = "minivecdb.db"
MODEL_CACHE_DIRNAME = "model_cache"
MODEL_CACHE_SUBDIR = "model_cache/huggingface"
DEFAULT_RUN_PREFIX = "demo"
PROJECT_ROOT_ENV = "MINIVECDB_PROJECT_ROOT"
```

---

## Functions (15 total)

### Path Resolution

#### `get_project_root() → str`
Returns the project root. Uses `MINIVECDB_PROJECT_ROOT` env var if set, otherwise resolves from this file's location (`core/` → parent).

#### `get_db_run_root(project_root=None) → str`
Returns `<project_root>/db_run/` path without creating it.

#### `ensure_db_run_root(project_root=None) → str`
Like `get_db_run_root` but creates the directory if missing. Returns absolute path.

#### `get_shared_db_path(project_root=None) → str`
Returns `<db_run>/minivecdb.db` absolute path. Creates `db_run/` if missing.

#### `get_model_cache_path(project_root=None) → str`
Returns `<db_run>/model_cache/huggingface/` absolute path. Creates the directory.

#### `is_within_db_run(path, project_root=None) → bool`
Returns `True` when `path` is located inside `db_run/`. Used by VectorStore to decide shared vs. per-folder DB routing.

---

### Active Run Management

#### `get_active_run_marker_path(project_root=None) → str`
Returns the path to `db_run/.active_run` file.

#### `read_active_run_path(project_root=None) → Optional[str]`
Reads `.active_run` file and validates:
1. File exists and is readable
2. Content is non-empty
3. Resolves to an absolute path (handles both relative and absolute values)
4. Path is not the model cache directory
5. Path is inside `db_run/`
6. Path is an existing directory

Returns `None` if any validation fails.

#### `write_active_run_name(run_name, project_root=None)`
Writes a run directory name (NOT path) to `.active_run`. Validates it's a simple name (no slashes).

#### `set_active_run_path(run_path, project_root=None)`
Marks a full path as active. Validates it's an existing directory inside `db_run/`. Extracts the basename and calls `write_active_run_name`.

---

### Run Directory Creation

#### `_sanitize_prefix(prefix) → str`
Strips non-alphanumeric characters from a prefix to make it filesystem-safe. Falls back to `"demo"` if empty.

#### `generate_run_name(prefix="demo") → str`
Generates a unique name: `<prefix>_<unix_timestamp>_<6-char-hex>`
Example: `demo_1765667200_a1b2c3`

#### `create_new_run_path(prefix="demo", project_root=None) → str`
Creates a new unique run directory under `db_run/`:
1. Generates a unique name via `generate_run_name`
2. Attempts `os.makedirs(exist_ok=False)` — retries up to 50 times on collision
3. Writes the new name to `.active_run`
4. Returns the absolute path

#### `resolve_storage_path(storage_path, create_new_run=False, run_prefix="demo", project_root=None) → str`
Top-level resolver used by VectorStore:
- If `storage_path` given → returns it as-is (absolute)
- If `create_new_run=True` → creates a fresh directory
- Else → reads `.active_run`; if empty, auto-creates a new run

---

### Run Directory Enumeration

#### `list_run_directories(project_root=None) → list`
Returns absolute paths of all run directories under `db_run/`, sorted by mtime (newest first). Skips hidden files and `model_cache/`.

---

## Design Notes

1. **No hardcoded paths**: Everything derives from `get_project_root()`. Override with `MINIVECDB_PROJECT_ROOT` env var.
2. **Retry loop**: `create_new_run_path` retries 50 times in the extremely unlikely case of uuid4 collision within the same timestamp.
3. **Validation chain**: `read_active_run_path` performs 6 checks before returning a path; any failure returns `None` rather than raising.
4. **Used by everyone**: VectorStore, CLI, web/app.py, migrations, and benchmarks all depend on this module.
