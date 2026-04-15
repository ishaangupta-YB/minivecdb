# MiniVecDB — File: `ARCHITECTURE.py` (v3.0)

> **Location**: `minivecdb/ARCHITECTURE.py`
> **Lines**: 526 | **Size**: 21 KB
> **Role**: Central specification — schema, queries, data models, ID generation

---

## What Changed from Earlier Versions

ARCHITECTURE.py has **grown from 152 lines to 526 lines** since the initial documentation. Major additions:

1. **6-table schema** with 3 triggers and 9 indexes (was in flux, now complete)
2. **35 parameterised SQL queries** in `SQL_QUERIES` dict (was partially defined)
3. **7 data models** as `@dataclass` objects (was 3)
4. **`generate_id()` function** for collision-resistant record IDs
5. **Self-test block** that exercises triggers, JOINs, aggregates, and cascades

---

## File Structure

The file is organized into 5 major sections:

```
Lines   1-18   : Module docstring (hybrid architecture overview)
Lines  19-21   : Imports (dataclasses, typing, numpy, time, uuid)
Lines  23-159  : SCHEMA_SQL constant (6 tables + 3 triggers)
Lines 161-325  : SQL_QUERIES dict (35 parameterised templates)
Lines 327-433  : Data models (7 dataclasses + generate_id)
Lines 436-452  : Disk layout documentation comment
Lines 455-526  : Self-test (__main__ block)
```

---

## SCHEMA_SQL Constant

A single multi-line SQL string containing all `CREATE TABLE`, `CREATE INDEX`, and `CREATE TRIGGER` statements. Executed via `conn.executescript(SCHEMA_SQL)` — safe to run multiple times (`IF NOT EXISTS` on every statement).

See [03_er_diagram_and_schema.md](./03_er_diagram_and_schema.md) for the full table/trigger breakdown.

---

## SQL_QUERIES Dict (35 Templates)

Every SQL query the application uses lives here. Never string-format SQL anywhere else. All use `?` parameterised placeholders.

### Session Queries (4)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `insert_session` | INSERT | Register a new session row |
| `upsert_session` | INSERT...ON CONFLICT DO UPDATE | Idempotent session registration |
| `get_session_by_name` | SELECT WHERE | Look up session by unique name |
| `touch_session` | UPDATE SET | Manually bump `last_used_at` |

### Session Aggregate (1)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `list_sessions_with_counts` | LEFT JOIN + GROUP BY + correlated subquery | Session picker: msg_count, record_count per session |

### Conversation Queries (2)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `get_default_conversation_for_session` | SELECT ORDER BY LIMIT 1 | First (default) conversation |
| `insert_conversation` | INSERT | Create a new conversation |

### Message Queries (2)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `insert_message` | INSERT (10 params) | Log a user query |
| `history_for_session` | INNER JOIN + ORDER BY + LIMIT | Chat history timeline |

### Collection Queries (6)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `create_collection` | INSERT | New collection in session |
| `get_collection_id_by_name` | SELECT WHERE (session_id, name) | Name → surrogate ID |
| `get_collection_full` | SELECT WHERE | Full collection row |
| `list_collections_in_session` | LEFT JOIN + GROUP BY | All collections with record counts |
| `delete_collection` | DELETE WHERE | Drop collection + cascade |
| `collection_exists` | SELECT 1 LIMIT 1 | Fast existence check |

### Record Queries (14)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `insert_record` | INSERT | Add record to session + collection |
| `get_record` | INNER JOIN (records ↔ collections) | Fetch by ID, scoped to session |
| `delete_record` | DELETE WHERE | Remove single record |
| `update_record_text` | UPDATE SET | Change text only |
| `list_records` | JOIN + ORDER BY + LIMIT | Records in a collection |
| `count_records` | COUNT + JOIN | Records in a collection (count) |
| `count_all_records` | COUNT WHERE | Total records in session |
| `record_exists` | SELECT 1 LIMIT 1 | Fast existence check |
| `all_record_ids` | SELECT ORDER BY | All IDs in session |
| `collection_record_ids` | JOIN + ORDER BY | IDs in one collection |
| `all_records_with_text` | SELECT WHERE | (id, text) pairs for rebuild |
| `delete_records_in_collection` | DELETE + scalar subquery | Clear one collection |
| `delete_all_records` | DELETE WHERE | Wipe session |
| `list_record_ids` / `list_record_ids_in_collection` | SELECT + LIMIT | Paginated ID listing |

### Metadata Queries (4)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `insert_metadata` | INSERT | Add key/value tag |
| `get_metadata` | SELECT WHERE | Retrieve all tags for record |
| `delete_metadata` | DELETE WHERE | Drop all tags for record |
| `filter_by_metadata` | JOIN (metadata ↔ records) + WHERE | Session-scoped metadata filter |

### Statistics Queries (1)
| Key | SQL Technique | Purpose |
|-----|--------------|---------|
| `stats_per_collection` | LEFT JOIN + GROUP BY | Collection → record count map |

---

## Data Models (7 Dataclasses)

### `VectorRecord`
```python
@dataclass
class VectorRecord:
    id: str                    # "vec_a1b2c3d4"
    vector: np.ndarray         # (384,) float32
    text: str
    metadata: Dict[str, Any]
    created_at: float
    collection: str = "default"
```
**New methods**: `to_dict()` for JSON serialization, `from_db_row(row, vector, metadata)` class method for hydration from SQLite.

### `SearchResult`
```python
@dataclass
class SearchResult:
    record: VectorRecord
    score: float
    rank: int
    metric: str
```
**New method**: `to_dict()` — returns `{id, text, metadata, score (rounded to 6dp), rank, metric}`.

### `CollectionInfo` (NEW)
```python
@dataclass
class CollectionInfo:
    name: str
    dimension: int
    count: int
    created_at: float
    description: str = ""
```

### `DatabaseStats` (UPDATED)
```python
@dataclass
class DatabaseStats:
    total_records: int
    total_collections: int
    dimension: int              # NEW
    memory_usage_bytes: int     # NEW
    storage_path: str           # NEW
    embedding_model: str        # NEW
    db_file: str                # NEW
    session_name: str = ""
```
Now includes all the information the web stats page needs: vector dimension, memory usage, model name, file paths.

### `SessionInfo`
```python
@dataclass
class SessionInfo:
    id: int
    name: str
    storage_path: str
    created_at: float
    last_used_at: float
    msg_count: int
    record_count: int
```

### `MessageRow`
```python
@dataclass
class MessageRow:
    id: int
    created_at: float
    kind: str                  # 'search' | 'insert'
    query_text: str
    metric: Optional[str]
    top_k: Optional[int]
    category_filter: Optional[str]
    result_count: Optional[int]
    elapsed_ms: Optional[float]
    response_ref: Optional[str]
```

---

## `generate_id()` Function

```python
def generate_id(prefix: str = "vec") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
```
Produces IDs like `vec_a1b2c3d4`. UUID4 ensures collision resistance.

---

## Self-Test Block (`if __name__ == "__main__"`)

**Lines 455–526**: Creates a temporary SQLite database, exercises every trigger and several JOIN/aggregate queries, and validates cascade deletes. Tests:

1. Insert session → trigger creates default conversation + default collection
2. Insert record via default collection → JOIN verify
3. Insert message → trigger bumps `last_used_at` (subquery)
4. `list_sessions_with_counts` → JOIN + GROUP BY aggregate
5. Delete session → cascade wipes all 5 child tables

All 5 assertions must pass or the schema is broken.
