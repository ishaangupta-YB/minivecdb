# MiniVecDB — File: core/vector_store.py (The Heart of the Project)

> **Location**: `minivecdb/core/vector_store.py`
> **Lines**: 1705 | **Size**: 66.2 KB
> **Purpose**: The main VectorStore class that integrates all three storage layers and implements the complete vector database API

---

## Why This File Exists

This is the **central nervous system** of MiniVecDB. Every user-facing operation (insert, search, get, delete, update, collections) flows through this class. It coordinates:

- **SQLite** (via `DatabaseManager`) for structured data
- **NumPy** (in-memory `_vectors` matrix) for vector math
- **JSON** (`id_mapping.json`) for the bridge between them
- **EmbeddingEngine** for text → vector conversion
- **Shared DB routing** (v3.0): Uses shared `db_run/minivecdb.db` when inside `db_run/`, otherwise per-folder DB for legacy/test compatibility
- **Session awareness** (v3.0): Every instance is bound to a `session_name` exposed as `self.session_name`

Think of it as the "engine" of a car: the wheels (CLI), steering (Web UI), and fuel (embeddings) all connect through it.

---

## Class: `VectorStore`

### Constructor: `__init__()`
**Lines 98–235**

The constructor does A LOT — it's the complete system bootstrap:

| Step | What It Does | Why |
|------|-------------|-----|
| 1 | Validate all arguments | Catch bad config early |
| 2 | Resolve storage path | Use explicit path, active run, or create new |
| 3 | Resolve model cache path | Where to store the 80MB embedding model |
| 4 | `os.makedirs(storage_path)` | Create storage directory |
| 5 | Open SQLite via `DatabaseManager` | Connect to `minivecdb.db` |
| 6 | Ensure collection exists | Create custom collection if needed |
| 7 | `create_embedding_engine(fallback=True)` | Load ML model or fallback |
| 8 | Initialize empty vectors | `np.empty((0, 384), dtype=float32)` |
| 9 | `_load_vectors()` | Restore from disk if previously saved |
| 10 | `_rebuild_id_index()` | Build O(1) lookup `{id: row_index}` |
| 11 | `_validate_internal_state()` | Final consistency check |

**Parameters**:
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `storage_path` | `None` | Explicit directory, or use managed db_run |
| `collection_name` | `"default"` | Default collection for operations |
| `dimension` | `384` | Must match embedding model |
| `embedding_model` | `None` | Override model name |
| `new_run` | `False` | Force a fresh run directory |
| `run_prefix` | `"demo"` | Prefix for run dir names |
| `model_cache_path` | `None` | Override model cache location |
| `session_name` | `None` | Explicit session name (v3.0); defaults to basename of storage_path |

**Shared DB routing** (lines 204–209): If `storage_path` is inside `db_run/`, the constructor calls `ensure_shared_db_exists()` to get the shared DB path at `db_run/minivecdb.db`. Otherwise, it creates a per-folder `minivecdb.db` alongside the vectors. This keeps existing test fixtures working.

**Session binding** (lines 211–224): The `session_name` parameter (or derived basename) is passed to `DatabaseManager`, which upserts the session row. The resulting `self.session_name` is used by the web UI and CLI.

**Legacy migration** (lines 263–312): If old-style folders (`minivecdb_data/`, `vectorstore_data/`) exist but no active run, the constructor automatically migrates them into `db_run/`.

---

### In-Memory Data Structures

```python
self._vectors: np.ndarray    # shape (N, 384), dtype=float32 — all embeddings
self._id_list: List[str]     # ["vec_001", "vec_002", ...] — ordered by row
self._id_to_index: Dict[str, int]  # {"vec_001": 0, "vec_002": 1} — O(1) lookup
```

**The Three-Way Bridge**:
- `_vectors[42]` = the 384-dim vector for the record at row 42
- `_id_list[42]` = `"vec_a1b2c3d4"` → the SQLite record ID at row 42
- `_id_to_index["vec_a1b2c3d4"]` = `42` → reverse lookup

---

## CRUD Operations

### `insert(text, metadata=None, id=None, collection=None) → str`
**Lines 380–528**

**What it does**: Adds one text record to the database.

**Step-by-step**:
1. Validate text is non-empty, collection exists
2. Generate unique ID if none provided
3. Check ID not duplicate (`db.record_exists(id)`)
4. Embed text → 384-dim float32 vector
5. **Atomic SQLite transaction**: insert record + metadata
6. Append vector to NumPy matrix: `np.vstack([_vectors, vector_2d])`
7. Append ID to `_id_list`, rebuild `_id_to_index`
8. `save()` to disk
9. **Rollback on failure**: if `save()` fails, revert in-memory state AND delete the SQLite record (compensating action)

**Why compensating delete?** SQLite auto-commits inside the transaction, but NumPy save might fail (disk full, permission error). To keep both systems consistent, we undo the SQLite insert.

---

### `insert_batch(texts, metadata_list=None, ids=None, collection=None) → List[str]`
**Lines 534–709**

**What it does**: Inserts multiple records at once — **10-50x faster** than calling `insert()` in a loop.

**Why faster?**:
1. `encode_batch(texts)` sends all texts through the neural network at once (GPU parallelism)
2. Only one `save()` call at the end instead of N calls
3. One SQLite transaction wraps all inserts

**Validation**: Checks ALL IDs for duplicates *before* doing any work, preventing half-inserted states.

---

### `get(id: str) → Optional[VectorRecord]`
**Lines 715–765**

**What it does**: Retrieves a complete record by reassembling data from three sources:

```python
row = self.db.get_record(id)          # SQLite: (id, text, collection, created_at)
metadata = self.db.get_metadata(id)   # SQLite: {"category": "science", ...}
vector = self._vectors[self._id_to_index[id]]  # NumPy: (384,) float32
return VectorRecord.from_db_row(row, vector, metadata)
```

**Returns** `None` if the ID doesn't exist.

---

### `delete(id: str) → bool`
**Lines 1314–1369**

**Step-by-step**:
1. Check existence
2. Find row index via `_id_to_index`
3. `db.delete_record(id)` — CASCADE auto-deletes metadata
4. `np.delete(_vectors, idx, axis=0)` — remove one row from matrix
5. `_id_list.pop(idx)` — remove from ID list
6. `_rebuild_id_index()` — all indices after `idx` shifted down
7. `save()` — persist changes

---

### `update(id, text=None, metadata=None) → bool`
**Lines 1375–1439**

**Two independent updates**:

1. **Text update** (if `text` is provided):
   - Re-embed the new text → new 384-dim vector
   - `db.update_record_text(id, text)` in SQLite
   - `_vectors[idx] = new_vector` — replace in-place (no vstack needed)

2. **Metadata update** (if `metadata` is provided):
   - Full replace: delete ALL existing metadata, then insert new pairs
   - This is simpler than diffing (delete "category" but keep "author")

---

## Search Operations

### `search(query, top_k=5, metric="cosine", filters=None, collection=None) → List[SearchResult]`
**Lines 1030–1168** | The core feature of a vector database.

**Algorithm**:

```
1. Validate inputs (database not empty, metric exists)
2. Embed query text → query_vector (384,)
3. Determine candidates:
   ├── If filters → _get_filtered_indices(filters, collection) → subset of rows
   ├── If collection only → get collection IDs → subset of rows
   └── If neither → all rows
4. Compute scores: batch_fn(query_vector, candidate_vectors) → (M,) scores
5. Sort: np.argsort(scores)[::-1] for higher-is-better, [:] for lower-is-better
6. Take top_k, build SearchResult objects with rank/score/record
```

**Pre-filtering**: The SQL metadata filter reduces the candidate set BEFORE computing expensive vector similarity. Searching 100 filtered records is much cheaper than searching 100,000.

### `search_by_vector(query_vector, top_k, metric, filters) → List[SearchResult]`
**Lines 1170–1260**

Same as `search()` but takes a pre-computed vector instead of text. Skips the embedding step. Useful for finding records similar to an existing record.

### `_get_filtered_indices(filters, collection) → np.ndarray`
**Lines 1266–1308**

The bridge between SQL-based metadata filtering and NumPy-based vector search:

```python
matching_ids = set(self.db.filter_by_metadata(filters))  # SQL query
if collection:
    col_ids = set(self.db.get_record_ids_in_collection(collection))
    matching_ids &= col_ids  # Intersection (AND logic)
indices = [self._id_to_index[rid] for rid in matching_ids]  # Convert to row indices
```

---

## Collection Management

### `create_collection(name, description="") → CollectionInfo`
**Lines 1451–1487**

Validates uniqueness, inserts into SQLite, returns info.

### `list_collections() → List[CollectionInfo]`
**Lines 1489–1515**

Uses a `LEFT JOIN` so collections with zero records still appear in results.

### `delete_collection(name) → bool`
**Lines 1517–1590**

1. Protects the "default" collection from deletion
2. Finds all record IDs in the collection
3. Deletes from SQLite (CASCADE removes records + metadata)
4. Removes corresponding vectors from NumPy matrix
5. Rebuilds index and saves

---

## Bulk Operations

### `list_ids(collection=None, limit=100) → List[str]`
Delegates to `db.list_record_ids()`.

### `clear(collection=None) → int`
**Lines 1619–1675**

Deletes all records (or all in a collection):
- Collection-specific: find IDs → delete from SQLite → remove vectors → save
- All records: `db.delete_all_records()` → reset to empty NumPy matrix → save

### `count(collection=None) → int`
Delegates to `db.count_records()`.

### `__len__() → int`
Enables `len(store)` syntax.

---

## Persistence

### `save() → str`
**Lines 781–831**

Writes two files atomically:
```python
# Write to temp files first
np.save("vectors.npy.tmp", _vectors)
json.dump(_id_list, "id_mapping.json.tmp")

# Atomic rename
os.replace("vectors.npy.tmp", "vectors.npy")
os.replace("vectors.npy.tmp", "id_mapping.json")
```

### `_load_vectors()`
**Lines 833–925**

Restores state from disk with extensive validation:
1. Both files must exist
2. `id_mapping.json` must be a valid JSON list of strings
3. Row count of `.npy` must match JSON list length
4. Every ID in mapping must exist in SQLite
5. If any check fails → `_rebuild_vectors()`

### `_rebuild_vectors()`
**Lines 927–973**

Emergency recovery: re-creates vectors from SQLite (the source of truth):
1. `db.get_all_records_with_text()` — get all IDs and texts
2. `embedding_engine.encode_batch(texts)` — re-embed everything
3. Replace `_vectors` and `_id_list`
4. `save()` — write repaired files

### `_validate_internal_state()`
**Lines 320–363**

Final consistency check after loading: verifies matrix dimensions, ID uniqueness, row count match with SQLite. If problems found, triggers one more rebuild.

---

## Context Manager

```python
with VectorStore("./data") as store:
    store.insert("Hello")
# close() called automatically here, even on exception
```

- `__enter__()` returns `self`
- `__exit__()` calls `close()` which calls `save()` then `db.close()`

---

## Static Helpers

| Method | Purpose |
|--------|---------|
| `_require_non_empty_string(value, name)` | Validation helper |
| `_normalize_metadata(metadata)` | Converts all values to strings |
| `_validate_top_k(top_k)` | Ensures positive integer |
| `_find_duplicate_ids(ids)` | Returns set of duplicated IDs |
| `_rebuild_id_index()` | Rebuilds `{id: row_index}` dict from `_id_list` |
| `_maybe_migrate_legacy_storage(prefix)` | Moves old-style data to db_run |

---

## Stats

### `stats() → DatabaseStats`
**Lines 1681–1718**

Returns overall statistics:
- `total_records`, `total_collections`, `dimension`
- `memory_usage_bytes`: N × 384 × 4 (each float32 = 4 bytes)
- `storage_path`, `embedding_model`, `db_file`, `session_name`

---

## v3.0 Key Imports

```python
from ARCHITECTURE import VectorRecord, SearchResult, CollectionInfo, DatabaseStats, generate_id
from storage.migrations import ensure_shared_db_exists
```

Notable additions:
- **`CollectionInfo`** — Used by `list_collections()` return values
- **`generate_id`** — Centralised ID generation (was previously inline)
- **`ensure_shared_db_exists`** — Ensures shared DB exists + runs legacy migrations
- **`is_within_db_run`** — Determines shared vs. per-folder DB routing
