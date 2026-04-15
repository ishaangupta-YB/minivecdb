# MiniVecDB — File: `storage/migrations.py`

> **Location**: `minivecdb/storage/migrations.py`
> **Lines**: 276 | **Size**: 9.6 KB
> **Role**: One-shot migration of legacy per-session SQLite files into the shared v3.0 DB

---

## Purpose

Before v3.0, each session had its own `minivecdb.db` inside its run folder. v3.0 consolidates all sessions into a single shared `db_run/minivecdb.db`. This module:

1. Detects pre-v3 per-session `.db` files
2. Copies their collections, records, and metadata into the shared DB under a new session row
3. Renames the old file to `minivecdb.db.legacy` (**never deletes user data**)
4. Is idempotent: safe to call repeatedly

---

## Constants

```python
LEGACY_DB_FILENAME = "minivecdb.db"
LEGACY_BACKUP_SUFFIX = ".legacy"
```

---

## Functions

### Detection Helpers

#### `_records_table_has_session_id(conn) → bool`
Checks if the `records` table has a `session_id` column using `PRAGMA table_info(records)`. If it does, the DB is already v3+ and doesn't need migration.

#### `_legacy_has_data(conn) → bool`
Returns `True` if the legacy DB has at least one record worth migrating (`COUNT(*) FROM records > 0`).

---

### Data Extraction Helpers

All read from the legacy DB opened in **read-only mode** (`?mode=ro`):

| Function | Returns | SQL |
|----------|---------|-----|
| `_legacy_collection_rows(conn)` | `[(name, dimension, description, created_at)]` | `SELECT ... FROM collections` |
| `_legacy_record_rows(conn)` | `[(id, text, collection, created_at)]` | `SELECT ... FROM records ORDER BY created_at ASC` |
| `_legacy_metadata_rows(conn)` | `[(record_id, key, value)]` | `SELECT ... FROM metadata` |

---

### Collision Handling

#### `_new_id_if_collision(shared, rid, session_name) → str`
Returns the `rid` unchanged if it's unique in the shared DB. Otherwise tries `{rid}_{session_name}`, then `{rid}_{session_name}_{microsecond_timestamp}`. This prevents duplicate PK errors when two legacy sessions had records with the same ID.

---

### Core Migration

#### `_migrate_one_session(shared, session_folder, legacy_db_path) → int`

Migrates one legacy per-session DB into the shared DB. Returns the number of records migrated (0 if already migrated or empty).

**Algorithm**:
1. Open legacy DB in read-only mode
2. Check: if `records` table has `session_id` column → already v3, skip (return 0)
3. Check: if no data → skip (return 0)
4. Read all collections, records, metadata from legacy
5. Wrap migration in explicit `BEGIN` / `COMMIT` transaction
6. Check: if session name already exists in shared DB → skip (return 0)
7. Insert session row (triggers auto-create default conversation + collection)
8. Map legacy collection names → new `collection_id` values
9. Remap record IDs if they collide with existing shared DB records
10. Insert all records with new session_id + collection_id
11. Insert all metadata with remapped record IDs
12. Commit transaction

**Error handling**: On any exception, rolls back the transaction and re-raises.

---

### Public API

#### `migrate_legacy_per_session_dbs(db_run_root) → int`

Scans `db_run_root` for all session folders containing a `minivecdb.db` file. For each:
1. Calls `_migrate_one_session()` to copy data into the shared DB
2. If records were migrated, renames the legacy file to `.legacy` (with timestamp suffix if `.legacy` already exists)
3. Returns total records migrated across all sessions

**Idempotency**: Already-migrated sessions are skipped (name check + file rename). Safe to call on every startup.

#### `ensure_shared_db_exists(db_run_root) → str`

High-level entry point:
1. Creates `db_run_root` directory if missing
2. Runs `migrate_legacy_per_session_dbs()` to handle any legacy DBs
3. If the shared DB doesn't exist yet, creates it with the v3 schema
4. Returns the absolute path to the shared DB file

**Used by**: `VectorStore.__init__()` and `web/app.py` on startup.

---

## Migration Safety Guarantees

| Concern | Safeguard |
|---------|-----------|
| Data loss | Legacy file is renamed to `.legacy`, never deleted |
| Repeated runs | Name-based skip + file rename = idempotent |
| Transaction atomicity | Explicit `BEGIN`/`COMMIT`/`ROLLBACK` per session |
| ID collisions | `_new_id_if_collision` generates unique suffixed IDs |
| Read-only legacy access | Opened with `?mode=ro` URI parameter |
| Vector files | Not touched — `vectors.npy` + `id_mapping.json` stay in place |
