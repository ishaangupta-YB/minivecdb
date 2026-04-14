# MiniVecDB — File: storage/database.py (SQLite Wrapper)

> **Location**: `minivecdb/storage/database.py`
> **Lines**: 786 | **Size**: 29.6 KB
> **Purpose**: The ONLY SQLite access layer — wraps all raw SQL behind clean Python methods (Repository Pattern)

---

## Why This File Exists

This module implements the **Repository Pattern** — it hides all raw SQL behind clean Python methods. The rest of the codebase (VectorStore, CLI, Web) never writes SQL directly; they call methods like `db.insert_record()` or `db.filter_by_metadata()`.

**Benefits**:
- SQL is centralised in one file (easier to audit, maintain, optimize)
- The rest of the code doesn't need to know about SQLite internals
- Changing the SQL implementation only requires changes here
- Parameterised queries prevent SQL injection everywhere automatically

---

## Class: `DatabaseManager`

### Constructor: `__init__(db_path: str)`
**Lines 65–105**

**Step-by-step**:
1. `sqlite3.connect(db_path, check_same_thread=False)`
   - Opens (or creates) the SQLite database file
   - `check_same_thread=False` allows Flask's multi-threaded request handling
2. `PRAGMA foreign_keys = ON`
   - **CRITICAL**: Without this, SQLite parses but IGNORES foreign key constraints
   - CASCADE deletes won't fire if this isn't set
3. `conn.executescript(SCHEMA_SQL)`
   - Runs all CREATE TABLE and CREATE INDEX statements
   - `IF NOT EXISTS` guards make it safe to run every time
4. `conn.commit()`
   - Finalises the schema creation

---

### Transaction Context Manager

#### `transaction() → Iterator[None]`
**Lines 113–133**

```python
@contextmanager
def transaction(self):
    try:
        self._conn.execute("BEGIN")
        yield
    except Exception:
        self._conn.rollback()
        raise
    else:
        self._conn.commit()
```

**What it does**: Groups multiple operations into one atomic transaction. If any operation raises an exception, ALL changes are rolled back.

**Used by**: `VectorStore.insert()` and `insert_batch()` for atomic record + metadata insertion:
```python
with self.db.transaction():
    self.db.insert_record(id, text, collection, created_at, auto_commit=False)
    for key, value in metadata.items():
        self.db.insert_metadata(id, key, value, auto_commit=False)
```

---

## Record CRUD Functions

### `insert_record(id, text, collection, created_at, auto_commit=True)`
**Lines 148–195**

Executes: `INSERT INTO records (id,text,collection,created_at) VALUES (?,?,?,?)`

**Why `auto_commit` parameter?**: When called inside a `transaction()`, we don't want each individual insert to commit — the transaction::commit() handles that. When called standalone, we commit immediately.

**Error handling**: Catches `sqlite3.IntegrityError` for:
- Duplicate primary key (record already exists)
- Foreign key violation (collection doesn't exist)

### `get_record(id) → Optional[Tuple]`
**Lines 197–217**

Returns `(id, text, collection, created_at)` tuple or `None`.

**Why tuple not dict?** SQLite's `fetchone()` returns tuples by default. Converting to a named structure would add overhead. The caller (VectorStore) knows the column order from the SELECT clause.

### `delete_record(id) → bool`
**Lines 219–243**

Uses `cursor.rowcount > 0` to determine if a record was actually deleted. CASCADE automatically deletes metadata rows.

### `update_record_text(id, new_text) → bool`
**Lines 245–271**

**Note**: This only updates the text in SQLite. The caller (VectorStore.update()) is responsible for re-embedding the vector.

**Subtle detail**: The SQL is `UPDATE records SET text=? WHERE id=?`, so `new_text` comes FIRST in the parameter tuple, then `id`.

### `record_exists(id) → bool`
**Lines 273–290**

Uses `SELECT 1 ... LIMIT 1` — faster than `get_record()` because it doesn't fetch actual columns, just checks existence.

---

## Metadata Operations (EAV Pattern)

### What Is EAV?
The metadata table uses **Entity-Attribute-Value** pattern:

| Column | EAV Role | Example |
|--------|----------|---------|
| `record_id` | Entity | `"vec_a1b2c3d4"` |
| `key` | Attribute | `"category"` |
| `value` | Value | `"science"` |

This allows each record to have completely different tags without schema changes.

### `insert_metadata(record_id, key, value, auto_commit=True)`
**Lines 311–347**

Adds one key-value tag row. Values are always stored as `str(value)`.

### `get_metadata(record_id) → Dict[str, str]`
**Lines 349–371**

Fetches all metadata for a record and converts from list of tuples to dict:
```python
# SQLite returns: [("category", "science"), ("author", "Einstein")]
# dict() converts: {"category": "science", "author": "Einstein"}
```

### `delete_metadata(record_id)`
**Lines 373–387**

Deletes ALL metadata for a record. Used by VectorStore.update() for "full replace" semantics.

### `filter_by_metadata(filters: Dict) → List[str]`
**Lines 402–452** | The metadata pre-filtering engine.

**Supports three filter types**:

#### Type 1: Exact Match (string value)
```python
{"category": "science"}
→ SELECT DISTINCT record_id FROM metadata WHERE key='category' AND value='science'
```

#### Type 2: List Match / OR (list of strings)
```python
{"category": ["science", "tech"]}
→ SELECT DISTINCT record_id FROM metadata WHERE key='category' AND value IN ('science','tech')
```

#### Type 3: Comparison Operators (dict)
```python
{"year": {"$gt": "2020", "$lte": "2025"}}
→ WHERE key='year' AND CAST(value AS REAL) > 2020 AND CAST(value AS REAL) <= 2025
```

**Supported operators** (`_FILTER_OPERATORS` dict, lines 394–400):
| Operator | SQL Expression | Example |
|----------|---------------|---------|
| `$gt` | `CAST(value AS REAL) > ?` | Year > 2020 |
| `$lt` | `CAST(value AS REAL) < ?` | Price < 50 |
| `$gte` | `CAST(value AS REAL) >= ?` | Score >= 0.8 |
| `$lte` | `CAST(value AS REAL) <= ?` | Count <= 100 |
| `$ne` | `value != ?` | Status != "draft" |

**AND logic**: Multiple filters are intersected: `result = set1 & set2 & set3`. A record must match ALL criteria.

**Why `CAST(value AS REAL)` for numeric operators?** All values are stored as TEXT. Without the cast, `"2021" > "2020"` would be a string comparison (which happens to work for numbers) but `"9" > "10"` would be wrong (string "9" > string "10"). Casting ensures numeric comparison.

### `_execute_single_filter(key, value) → set`
**Lines 454–536** | Internal method that handles one filter criterion. Uses `isinstance()` checks to determine which SQL to generate.

---

## Record Listing & ID Retrieval

### `get_record_ids_in_collection(collection) → List[str]`
**Lines 542–558** | Returns all IDs in a collection, ordered by creation time.

### `get_all_record_ids() → List[str]`
**Lines 560–569** | Returns every record ID across all collections.

### `get_all_records_with_text() → List[Tuple[str, str]]`
**Lines 571–582** | Returns `(id, text)` pairs for ALL records. Used by `_rebuild_vectors()` for emergency re-embedding.

### `list_record_ids(collection=None, limit=100) → List[str]`
**Lines 584–605** | Paginated ID listing with optional collection filter.

### `delete_records_in_collection(collection) → int`
**Lines 607–624** | Bulk delete all records in a collection. Returns count deleted.

### `delete_all_records() → int`
**Lines 626–637** | Nuclear option: delete ALL records across all collections.

---

## Collection CRUD

### `create_collection(name, dimension=384, description="")`
**Lines 648–677**

Inserts into the `collections` table. Raises `ValueError` if the name already exists (PRIMARY KEY violation).

### `list_collections() → List[Tuple]`
**Lines 679–693**

Uses a `LEFT JOIN` between collections and records to compute per-collection record counts even for empty collections:
```sql
SELECT c.name, c.dimension, c.description, c.created_at, COUNT(r.id) as cnt
FROM collections c LEFT JOIN records r ON c.name = r.collection
GROUP BY c.name ORDER BY c.created_at
```

### `delete_collection(name) → bool`
**Lines 695–714**

CASCADE delete: removing a collection automatically removes all its records (which in turn removes their metadata).

### `collection_exists(name) → bool`
**Lines 716–730**

Uses `SELECT 1 ... LIMIT 1` for fast existence check.

---

## Statistics

### `count_records(collection=None) → int`
**Lines 736–757**

Uses `SELECT COUNT(*)` — two variants:
- All records: `count_all_records` query (no WHERE)
- Per-collection: `count_records` query (WHERE collection=?)

### `stats_per_collection() → Dict[str, int]`
**Lines 759–772**

Group-by query that returns `{"default": 42, "science": 15}`.

---

## Connection Management

### `close()`
**Lines 778–785**

Closes the SQLite connection and releases the file lock. Must be called when done — `VectorStore.close()` calls this.

---

## Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Repository** | Entire class | Encapsulates all SQL behind clean methods |
| **Context Manager** | `transaction()` | Atomic multi-step writes with rollback |
| **Strategy** | `_execute_single_filter()` | Different SQL for str/list/dict values |
| **Parameterised Queries** | Every SQL call | Prevents SQL injection |
