# MiniVecDB — File: core/runtime_paths.py

> **Location**: `minivecdb/core/runtime_paths.py`
> **Lines**: 199 | **Size**: 6.8 KB
> **Purpose**: Centralises all path management for db_run directories, active run tracking, and model cache

---

## Why This File Exists

MiniVecDB stores runtime artifacts (SQLite databases, vector files, model cache) in a structured directory tree under `db_run/`. This module provides a single, consistent API for:

1. **Resolving where to store data** — explicit path or managed run
2. **Tracking the "active run"** — so CLI commands reuse the same data directory
3. **Creating unique run names** — `demo_<timestamp>_<random>` prevents collisions
4. **Locating the model cache** — keeps 80MB of model files in the project

Without this module, every file would independently construct paths, leading to drift and inconsistency.

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `DB_RUN_DIRNAME` | `"db_run"` | Name of the root runtime directory |
| `ACTIVE_RUN_FILENAME` | `".active_run"` | Marker file containing current run name |
| `MODEL_CACHE_DIRNAME` | `"model_cache"` | Parent of the HuggingFace cache |
| `MODEL_CACHE_SUBDIR` | `"model_cache/huggingface"` | Full subpath for model weights |
| `DEFAULT_RUN_PREFIX` | `"demo"` | Default prefix for run directory names |
| `PROJECT_ROOT_ENV` | `"MINIVECDB_PROJECT_ROOT"` | Optional env var to override root detection |

---

## Functions — Detailed Breakdown

### `_sanitize_prefix(prefix: str) → str`
**Line 27** | Internal helper that strips unsafe characters from a run name prefix.

Uses `re.sub(r"[^A-Za-z0-9_-]+", "_", prefix)` to replace anything that isn't alphanumeric, underscore, or hyphen. Falls back to `"demo"` if the result is empty.

**Why?** The prefix becomes part of a directory name. Characters like `/`, `\`, or spaces would break filesystem operations.

---

### `get_project_root() → str`
**Lines 33–43** | Determines the project root directory.

**Algorithm**:
1. Check if `MINIVECDB_PROJECT_ROOT` environment variable is set → use that
2. Otherwise: this file is at `core/runtime_paths.py` → go two directories up → that's the project root

**Why env var?** In testing or deployment, the working directory might not be the project root. The env var gives an explicit override.

---

### `get_db_run_root(project_root=None) → str`
**Lines 46–49** | Returns the absolute path to `<project_root>/db_run/`.

### `ensure_db_run_root(project_root=None) → str`
**Lines 52–56** | Same as above, but creates the directory if it doesn't exist.

---

### `get_active_run_marker_path(project_root=None) → str`
**Lines 59–64** | Returns path to `db_run/.active_run`.

### `is_within_db_run(path, project_root=None) → bool`
**Lines 67–71** | Security check: ensures a path is inside `db_run/`. Uses `os.path.commonpath()` to prevent directory traversal attacks.

---

### `read_active_run_path(project_root=None) → Optional[str]`
**Lines 74–109** | Reads the `.active_run` marker file and validates the run directory.

**Algorithm**:
1. Check if `.active_run` file exists → None if not
2. Read the file content (e.g., `"demo_1713052800_a1b2c3"`)
3. Resolve to absolute path (handling both relative and absolute values)
4. Validate:
   - Not the `model_cache` directory
   - Inside `db_run/` (security)
   - Actually exists on disk
5. Return the absolute path, or None if any check fails

**Why so many checks?** The marker file could be manually edited or corrupted. Defensive validation prevents the system from using a bad path.

---

### `write_active_run_name(run_name, project_root=None)`
**Lines 112–119** | Writes a run directory name to `.active_run`.

Validates the name doesn't contain path separators (must be a simple directory name, not a path). Writes with a trailing newline.

### `set_active_run_path(run_path, project_root=None)`
**Lines 122–137** | Higher-level function that validates a full path and writes just the basename to the marker.

---

### `generate_run_name(prefix="demo") → str`
**Lines 140–149** | Creates a unique run directory name.

**Format**: `<prefix>_<unix_timestamp>_<6_hex_chars>`
**Example**: `demo_1713052800_a1b2c3`

Uses both timestamp (for human-readability) and UUID (for uniqueness if multiple runs start in the same second).

---

### `create_new_run_path(prefix="demo", project_root=None) → str`
**Lines 152–170** | Creates a new unique run directory and marks it active.

**Algorithm**:
1. Loop up to 50 times (in case of collisions):
   - Generate a unique name
   - Try `os.makedirs(path, exist_ok=False)` — fails if dir already exists
   - If created successfully, write to `.active_run` and return the path
2. If 50 attempts all collide, raise `RuntimeError`

**Why `exist_ok=False`?** This ensures atomicity — only one process can create a specific directory name. If two processes race, one will get a `FileExistsError` and retry with a different name.

---

### `resolve_storage_path(storage_path, create_new_run, run_prefix, project_root) → str`
**Lines 173–190** | The main path resolution function — called by `VectorStore.__init__()`.

**Decision tree**:
```
if storage_path is not None:
    → Use it directly (explicit path mode)
elif create_new_run:
    → create_new_run_path() (force fresh run)
elif active run exists:
    → Use the active run (reuse last session)
else:
    → create_new_run_path() (first-time startup)
```

This is the function that implements the "default behavior" described in the AGENTS.md: if no path is specified, reuse the active run.

---

### `get_model_cache_path(project_root=None) → str`
**Lines 193–198** | Returns (and creates) the Hugging Face model cache directory.

Path: `db_run/model_cache/huggingface/`

The embedding engine uses this to cache the ~80MB `all-MiniLM-L6-v2` model weights locally, so they're only downloaded once.
