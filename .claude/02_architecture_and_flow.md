# MiniVecDB — Architecture & Data Flow

## High-Level Architecture

MiniVecDB follows a **layered hybrid architecture** where three storage engines work together, coordinated by the `VectorStore` class:

```
┌─────────────────────────────────────────────────┐
│                   USER INTERFACE                 │
│           CLI (argparse) / Web (Flask)           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              VectorStore (core engine)            │
│  insert() / search() / get() / delete() / etc.  │
└──┬──────────────┬──────────────────┬─────────────┘
   │              │                  │
   ▼              ▼                  ▼
┌────────┐  ┌──────────┐  ┌──────────────────────┐
│ SQLite │  │  NumPy   │  │  EmbeddingEngine     │
│ (DB)   │  │ (.npy)   │  │  (text → vectors)    │
└────────┘  └──────────┘  └──────────────────────┘
   │              │
   │    ┌─────────┘
   ▼    ▼
┌──────────────────┐
│  id_mapping.json │  ← Bridge file
│  (row ↔ ID)      │
└──────────────────┘
```

---

## Layer Descriptions

### Layer 1: User Interface (CLI / Web)

| Component | File | Purpose |
|-----------|------|---------|
| CLI | `cli/main.py` | Terminal commands via argparse (insert, search, get, delete, etc.) |
| Web | `web/app.py` | Session-aware browser UI (session picker → search/insert/history/stats) |
| Web API | `web/app.py` (`/api/search`) | JSON endpoint for programmatic access |

Both layers are thin wrappers — they parse user input, call VectorStore methods, and format output. **No business logic lives here.** The web layer adds session management (binding/switching VectorStore to different sessions).

### Layer 2: Core Engine (VectorStore)

The `VectorStore` class in `core/vector_store.py` is the **heart of MiniVecDB**. It coordinates all three storage layers:

- Calls `EmbeddingEngine.encode()` to convert text → vectors
- Calls `DatabaseManager` methods for SQL operations
- Manages the in-memory NumPy matrix directly
- Handles persistence (save/load of `.npy` and `.json` files)
- Implements search logic (pre-filter → batch similarity → ranking)

### Layer 3: Storage Engines

| Engine | File | Stores | Access Pattern |
|--------|------|--------|---------------|
| **SQLite** | `storage/database.py` | Sessions, Conversations, Messages, Collections, Records, Metadata | SQL queries via `DatabaseManager` (session-scoped) |
| **NumPy** | In-memory `_vectors` matrix, persisted to `vectors.npy` | 384-dim float32 vectors | Direct array indexing + batch operations |
| **JSON Bridge** | `id_mapping.json` | Ordered list mapping row index → record ID | Direct file I/O |
| **Migrations** | `storage/migrations.py` | Legacy per-session DB → shared DB migration | One-shot on startup |

### Layer 4: Embedding Engine

| Engine | File | Model | Quality |
|--------|------|-------|---------|
| `EmbeddingEngine` | `core/embeddings.py` | `all-MiniLM-L6-v2` (384-dim) | High quality (neural) |
| `SimpleEmbeddingEngine` | `core/embeddings.py` | Bag-of-words (hash-based) | Low quality (fallback) |

---

## Data Flow Diagrams

### INSERT Flow

When `store.insert("Python is great", metadata={"topic": "programming"})` is called:

```
1. Input validation
   ├── Check text is non-empty string
   ├── Validate collection exists
   └── Check ID not duplicate

2. Text → Vector conversion
   └── EmbeddingEngine.encode("Python is great")
       → np.array([0.12, -0.03, ...], dtype=float32)  # shape (384,)

3. SQLite storage (atomic transaction)
   ├── INSERT INTO records (id, text, collection, created_at)
   └── INSERT INTO metadata (record_id, "topic", "programming")

4. NumPy storage (in-memory)
   ├── np.vstack([_vectors, new_vector])  # (N,384) → (N+1,384)
   └── _id_list.append("vec_a1b2c3d4")

5. Disk persistence
   ├── np.save("vectors.npy", _vectors)
   └── json.dump(_id_list, "id_mapping.json")

6. Return record ID → "vec_a1b2c3d4"
```

### SEARCH Flow

When `store.search("artificial intelligence", top_k=3, metric="cosine", filters={"year": {"$gt": "2020"}})` is called:

```
1. Input validation
   ├── Check database not empty
   └── Validate metric name

2. Embed the query
   └── EmbeddingEngine.encode("artificial intelligence")
       → query_vector: shape (384,)

3. Pre-filter candidates (SQL)
   ├── SELECT record_id FROM metadata WHERE key='year' AND CAST(value AS REAL) > 2020
   ├── Intersect with collection IDs (if collection specified)
   └── Convert matching IDs → NumPy row indices

4. Batch similarity computation (NumPy)
   ├── candidate_vectors = _vectors[filtered_indices]  # shape (M, 384)
   ├── scores = batch_cosine_similarity(query_vector, candidate_vectors)  # shape (M,)
   └── This uses: candidate_vectors @ query_vector (matrix × vector)

5. Ranking
   ├── sorted_positions = np.argsort(scores)[::-1]  # descending
   └── top_k_positions = sorted_positions[:3]

6. Build results
   ├── For each top result:
   │   ├── Look up record ID from _id_list
   │   ├── Fetch record from SQLite (text, metadata)
   │   └── Build SearchResult(record, score, rank, metric)
   └── Return [SearchResult, SearchResult, SearchResult]
```

### GET Flow

When `store.get("vec_a1b2c3d4")` is called:

```
1. Fetch from SQLite
   ├── SELECT id, text, collection, created_at FROM records WHERE id=?
   └── SELECT key, value FROM metadata WHERE record_id=?

2. Fetch vector from NumPy
   ├── idx = _id_to_index["vec_a1b2c3d4"]  → 42
   └── vector = _vectors[42]  → shape (384,)

3. Build VectorRecord
   └── VectorRecord(id, vector, text, metadata, created_at, collection)
```

### DELETE Flow

When `store.delete("vec_a1b2c3d4")` is called:

```
1. Check existence
   └── SELECT 1 FROM records WHERE id=? LIMIT 1

2. Find row index
   └── idx = _id_to_index["vec_a1b2c3d4"]  → 42

3. Delete from SQLite
   └── DELETE FROM records WHERE id=?
       (CASCADE auto-deletes metadata rows)

4. Delete from NumPy
   ├── _vectors = np.delete(_vectors, 42, axis=0)  # (N,384) → (N-1,384)
   └── _id_list.pop(42)

5. Rebuild index mapping
   └── _id_to_index = {id: idx for idx, id in enumerate(_id_list)}

6. Persist to disk
   ├── np.save("vectors.npy", _vectors)
   └── json.dump(_id_list, "id_mapping.json")
```

---

## Startup / Initialization Flow

When `VectorStore(storage_path=None)` is instantiated:

```
1. Resolve storage path
   ├── Check if --db-path was given explicitly
   ├── Else check .active_run marker for last-used run
   ├── Else try migrating legacy folders (minivecdb_data/, vectorstore_data/)
   └── Else create new run directory: db_run/demo_<timestamp>_<random>/

2. Create storage directory
   └── os.makedirs(storage_path, exist_ok=True)

3. Open SQLite connection (v3.0: shared DB routing)
   ├── If storage path is inside db_run/:
   │   └── Use SHARED DB at db_run/minivecdb.db (ensure_shared_db_exists)
   ├── Else (test/legacy path):
   │   └── Use per-folder DB at <storage_path>/minivecdb.db
   ├── DatabaseManager binds to session (upsert session row)
   │   └── Triggers auto-create default conversation + collection
   ├── PRAGMA foreign_keys = ON
   └── Execute SCHEMA_SQL (CREATE TABLE IF NOT EXISTS ...)

4. Ensure collection exists
   └── CREATE collection if specified and doesn't exist

5. Initialize embedding engine
   └── create_embedding_engine(fallback=True)
       ├── Try: EmbeddingEngine (sentence-transformers)
       └── Else: SimpleEmbeddingEngine (bag-of-words)

6. Load vectors from disk
   ├── Load vectors.npy → _vectors (N, 384)
   ├── Load id_mapping.json → _id_list
   ├── If files missing/corrupt → _rebuild_vectors()
   │   └── Re-embed all texts from SQLite
   ├── Validate consistency (row count, ID overlap)
   └── Build _id_to_index lookup dict
```

---

## Persistence Model

### What Gets Persisted Where

| Data | Storage | When Persisted | Recovery |
|------|---------|---------------|----------|
| Records (id, text, collection) | SQLite shared `minivecdb.db` | On every insert/update/delete (auto-commit) | N/A (authoritative source) |
| Metadata (key, value) | SQLite shared `minivecdb.db` | On every insert/update/delete (auto-commit) | N/A (authoritative source) |
| Collections | SQLite shared `minivecdb.db` | On create/delete | N/A (authoritative source) |
| Sessions, Conversations | SQLite shared `minivecdb.db` | On session creation/switch | N/A (authoritative source) |
| Messages (chat history) | SQLite shared `minivecdb.db` | On every search/insert via `log_message()` | N/A (authoritative source) |
| Vectors (N × 384 matrix) | Per-session `vectors.npy` | After every insert/update/delete via `save()` | `_rebuild_vectors()` re-embeds from SQLite |
| ID mapping (row → ID) | Per-session `id_mapping.json` | After every insert/update/delete via `save()` | `_rebuild_vectors()` rebuilds from SQLite |
| Active run pointer | `.active_run` | On new run creation or session switch | Falls back to creating new run |

### Crash Recovery

SQLite is the **source of truth**. If `vectors.npy` or `id_mapping.json` are corrupted or missing:

1. `_load_vectors()` detects the inconsistency
2. `_rebuild_vectors()` is called
3. All texts are fetched from SQLite
4. Texts are re-embedded through the embedding engine
5. New `.npy` and `.json` files are written

This is slow (re-embeds everything) but guarantees correctness.

### Atomic Saves

The `save()` method uses a **write-to-temp-then-rename** pattern:

```python
# Write to temp files first
np.save("vectors.npy.tmp", _vectors)
json.dump(_id_list, "id_mapping.json.tmp")

# Atomic rename (can't partially fail)
os.replace("vectors.npy.tmp", "vectors.npy")
os.replace("id_mapping.json.tmp", "id_mapping.json")
```

If the process crashes between the two renames, at worst one file is stale, and the next startup detects the inconsistency and triggers a rebuild.

---

## The Three-Way Bridge

The fundamental challenge is linking data across three systems:

```
SQLite:     records table has "vec_a1b2c3d4" at row id='vec_a1b2c3d4'
NumPy:      _vectors has the 384-dim array at row index 42
JSON:       _id_list[42] = "vec_a1b2c3d4"
                └── This is the BRIDGE

_id_to_index = {"vec_a1b2c3d4": 42}   ← Reverse lookup (O(1) by ID)
```

| Lookup Direction | Mechanism | Time Complexity |
|-----------------|-----------|----------------|
| ID → Vector | `_id_to_index[id]` → `_vectors[idx]` | O(1) |
| Row index → ID | `_id_list[idx]` | O(1) |
| ID → Record data | `db.get_record(id)` | O(1) via PRIMARY KEY |
| Metadata filter → IDs | `db.filter_by_metadata(filters)` | O(n) scan or O(log n) with index |

---

## Threading & Concurrency Model

MiniVecDB uses **single-writer** semantics:
- SQLite is opened with `check_same_thread=False` to support Flask's multi-threaded request handling
- NumPy operations are not thread-safe — concurrent writes to `_vectors` would corrupt data
- For a learning project, this is acceptable. Production databases use locks, MVCC, or per-thread copies.
