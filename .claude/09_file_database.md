# MiniVecDB — File: `storage/database.py` (v3.0)

> **Location**: `minivecdb/storage/database.py`
> **Lines**: 565 | **Size**: 21.5 KB
> **Role**: Session-bound SQLite access layer — all record/metadata/collection/message CRUD

---

## What Changed from Earlier Versions

The DatabaseManager has been **substantially rewritten** from 786→565 lines (more focused, less duplication):

1. **Session-bound design**: Every instance is locked to `self.session_id` at construction
2. **Shared DB support**: Multiple sessions share one `.db` file; FK isolation at the row level
3. **New v3.0 methods**: `list_sessions_with_counts()`, `log_message()`, `get_history()`
4. **Backward compatibility**: Constructor auto-derives `session_name` from `db_path` for legacy callers
5. **All SQL comes from `ARCHITECTURE.py`**: Zero inline SQL construction (except `_execute_single_filter` for dynamic operators)

---

## Constructor

```python
DatabaseManager(db_path: str, session_name: Optional[str] = None, session_storage_path: Optional[str] = None)
```

Construction sequence:
1. Creates parent directory for `db_path` if missing
2. Opens SQLite connection with `check_same_thread=False`
3. Enables `PRAGMA foreign_keys = ON`
4. Executes `SCHEMA_SQL` (idempotent: `IF NOT EXISTS`)
5. Derives `session_name` from parent dir if not provided (backward compat)
6. Upserts session row → triggers auto-create default conversation + collection
7. Caches `self.session_id` and `self.conversation_id`

**Key attributes**:
- `self.db_path: str` — Path to the SQLite file
- `self.session_name: str` — Name of the bound session
- `self.session_id: int` — Cached session row ID
- `self.conversation_id: int` — Cached default conversation ID

---

## Internal Helpers

### `_require_non_empty_string(value, field_name)` — static
Raises `ValueError` if the value is not a non-empty string. Used as input validation throughout.

### `_ensure_session(name, storage_path) → int`
Upserts a session row using `SQL_QUERIES["upsert_session"]` and returns the session ID. The UPSERT ensures idempotency: calling with the same name just updates `storage_path`.

### `_get_or_create_default_conversation(session_id) → int`
Returns the first conversation's ID for the session. Usually the trigger already created it; if not (edge case), inserts one manually.

### `_resolve_collection_id(name) → int`
Looks up the collection's surrogate INTEGER PK from the composite `(session_id, name)` key. Raises `ValueError` if the collection doesn't exist.

### `transaction() → Iterator[None]`
Context manager for atomic multi-write operations:
```python
with dm.transaction():
    dm.insert_record(...)
    dm.insert_metadata(...)
    # Committed atomically. Rolled back on exception.
```

---

## Record CRUD (Session-Scoped)

Every record operation is automatically scoped to `self.session_id`.

### `insert_record(id, text, collection, created_at, auto_commit=True)`
- Resolves `collection` name → `collection_id` via `_resolve_collection_id`
- Inserts into `records` with `(id, session_id, collection_id, text, created_at)`
- `auto_commit=True` by default; set `False` inside `transaction()` blocks

### `get_record(id) → Optional[Tuple]`
Returns `(id, text, collection_name, created_at)` via a JOIN on records ↔ collections. Returns `None` if not found in the bound session.

### `delete_record(id) → bool`
Deletes the record (scoped to session). Cascades to metadata. Returns `True` if a row was deleted.

### `update_record_text(id, new_text) → bool`
Updates only the text column. The caller (VectorStore) is responsible for re-embedding and updating the vector.

### `record_exists(id) → bool`
Fast `SELECT 1 ... LIMIT 1` existence check scoped to the session.

---

## Metadata (EAV Pattern)

### `insert_metadata(record_id, key, value, auto_commit=True)`
Attaches a key/value tag to a record.

### `get_metadata(record_id) → Dict[str, str]`
Returns `{key: value}` dict for a record.

### `delete_metadata(record_id)`
Drops all metadata rows for a record.

### `filter_by_metadata(filters) → List[str]`
Returns record IDs matching ALL filter conditions (AND logic). Supports three value types:

| Value Type | Behavior | Example |
|-----------|----------|---------|
| `str` | Exact match | `{"category": "Science"}` |
| `list` | IN-set match | `{"category": ["Science", "Health"]}` |
| `dict` | Operator comparison | `{"year": {"$gt": 2020}}` |

**Supported operators** (`_FILTER_OPERATORS`):
- `$gt` / `$lt` / `$gte` / `$lte` — Numeric comparison (casts TEXT to REAL)
- `$ne` — Not equal (string comparison)

All queries scope via `JOIN records r ON r.id = m.record_id WHERE r.session_id = ?`.

---

## Record Listing & ID Retrieval

| Method | Returns | Query Used |
|--------|---------|-----------|
| `get_record_ids_in_collection(collection)` | `List[str]` | `collection_record_ids` |
| `get_all_record_ids()` | `List[str]` | `all_record_ids` |
| `get_all_records_with_text()` | `List[Tuple[str, str]]` | `all_records_with_text` |
| `list_record_ids(collection=None, limit=100)` | `List[str]` | `list_record_ids` or `list_record_ids_in_collection` |
| `delete_records_in_collection(collection) → int` | Deleted count | `delete_records_in_collection` (scalar subquery) |
| `delete_all_records() → int` | Deleted count | `delete_all_records` |

---

## Collection CRUD (Session-Scoped)

### `create_collection(name, dimension=384, description="")`
Inserts into `collections` with `(session_id, name, dimension, description, time.time())`. The composite UNIQUE constraint prevents duplicate names within a session.

### `list_collections() → List[Tuple]`
Returns `(name, dimension, description, created_at, record_count)` tuples. Uses a LEFT JOIN + GROUP BY query internally. The return shape preserves backward compatibility with pre-v3 callers.

### `delete_collection(name) → bool`
Drops a collection and cascades to its records + metadata.

### `collection_exists(name) → bool`
Fast existence check using `SELECT 1 LIMIT 1`.

---

## Statistics

### `count_records(collection=None) → int`
Total records in the session, or in one collection if specified.

### `stats_per_collection() → Dict[str, int]`
Returns `{collection_name: record_count}` for the bound session. Uses LEFT JOIN + GROUP BY.

---

## Sessions / Conversations / Messages (v3.0)

### `list_sessions_with_counts() → List[SessionInfo]`
Executes the `list_sessions_with_counts` query (LEFT JOIN sessions → conversations → messages, plus a correlated subquery for record count). Returns `SessionInfo` dataclass instances sorted by `last_used_at DESC`.

### `log_message(kind, query_text, *, metric, top_k, category_filter, result_count, elapsed_ms, response_ref, conversation_id) → int`
Persists one chat-history row. Validates `kind ∈ {'search', 'insert'}`. Defaults to `self.conversation_id` if `conversation_id` is not provided. The trigger automatically bumps `sessions.last_used_at`.

### `get_history(limit=200) → List[MessageRow]`
Returns chronological message list for the bound session. Uses the `history_for_session` JOIN query (messages → conversations → filter by session_id).

---

## Connection Management

### `close()`
Closes the underlying SQLite connection. Should always be called when done.

---

## Design Decisions

1. **Session binding at construction**: Rather than passing `session_id` to every method, it's cached once. This matches the architecture: a VectorStore instance is always tied to one session.

2. **SQL from ARCHITECTURE.py**: All 35 queries come from the central `SQL_QUERIES` dict. The only dynamic SQL is in `_execute_single_filter` for operator-based metadata filters.

3. **auto_commit flag**: Record and metadata inserts can defer commits for batch operations. The `transaction()` context manager provides explicit SAVEPOINT-like semantics.

4. **Backward compatibility**: When no `session_name` is provided, the manager derives it from the parent directory of `db_path`, so test fixtures like `DatabaseManager("/tmp/foo/minivecdb.db")` continue working without changes.
