# MiniVecDB — File: ARCHITECTURE.py (Central Specification)

> **Location**: `minivecdb/ARCHITECTURE.py` (project root)
> **Lines**: 152 | **Size**: 8.9 KB
> **Purpose**: The single source of truth for data models, SQL schema, and query templates

---

## Why This File Exists

`ARCHITECTURE.py` is the **central design specification** for the entire project. Instead of scattering data models and SQL strings across multiple files, everything is defined here and imported by other modules. This follows a key principle: **define once, use everywhere**.

Every module that touches the database imports from here:
- `core/vector_store.py` imports `VectorRecord`, `SearchResult`, `CollectionInfo`, `DatabaseStats`, `generate_id`
- `storage/database.py` imports `SCHEMA_SQL`, `SQL_QUERIES`

---

## Section Breakdown

### 1. SQL Schema (`SCHEMA_SQL`)
**Lines 20–48** — A multi-statement SQL string that creates all 3 tables + indexes.

```python
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS collections (...);
INSERT OR IGNORE INTO collections (...);  -- Auto-create 'default' collection
CREATE TABLE IF NOT EXISTS records (...);
CREATE INDEX IF NOT EXISTS idx_records_collection ON records(collection);
CREATE TABLE IF NOT EXISTS metadata (...);
CREATE INDEX IF NOT EXISTS idx_metadata_kv ON metadata(key, value);
CREATE INDEX IF NOT EXISTS idx_metadata_record ON metadata(record_id);
"""
```

**Why here?** The schema is a contract. Defining it centrally means:
- Any developer can see the exact table structure by reading one file
- Changes to the schema only need to happen in one place
- `IF NOT EXISTS` guards make it safe to run on every startup

### 2. SQL Queries (`SQL_QUERIES`)
**Lines 50–76** — A dictionary mapping human-readable names to parameterised SQL strings.

```python
SQL_QUERIES = {
    "insert_record": "INSERT INTO records (id,text,collection,created_at) VALUES (?,?,?,?)",
    "get_record": "SELECT id,text,collection,created_at FROM records WHERE id=?",
    ...
}
```

**Why a dictionary?** Instead of writing SQL inline in `database.py`, all queries live here:
- Easy to audit: "what SQL does this project run?" — look at one dict
- Prevents typos: `SQL_QUERIES["insert_record"]` vs writing the SQL from memory every time
- Parameterised `?` placeholders prevent SQL injection (never f-strings with SQL)

### 3. Data Models (Dataclasses)
**Lines 81–105** — Four `@dataclass` definitions that represent the project's core data structures.

#### `VectorRecord` (Line 82)
```python
@dataclass
class VectorRecord:
    id: str                    # "vec_a1b2c3d4"
    vector: np.ndarray         # shape (384,) float32
    text: str                  # "The cat sat on the mat"
    metadata: Dict[str, Any]   # {"category": "animals"}
    created_at: float          # 1713052800.0 (Unix timestamp)
    collection: str = "default"
```

**What it is**: The complete representation of one record, assembled from all three storage layers:
- `id`, `text`, `collection`, `created_at` come from SQLite `records` table
- `metadata` comes from SQLite `metadata` table
- `vector` comes from the NumPy `_vectors` matrix

**Methods**:
- `to_dict()` — Serializes the record (excluding the vector, which is too large for JSON)
- `from_db_row(row, vector, metadata)` — Factory method that builds a VectorRecord from a SQLite row tuple, a NumPy vector, and a metadata dict

#### `SearchResult` (Line 91)
```python
@dataclass
class SearchResult:
    record: VectorRecord       # The matched record
    score: float               # 0.9234 (similarity score)
    rank: int                  # 1 = best match
    metric: str                # "cosine" | "euclidean" | "dot"
```

**What it is**: One search result, wrapping a VectorRecord with ranking info.

**Methods**:
- `to_dict()` — Serializes for JSON output (rounds score to 6 decimals)

#### `CollectionInfo` (Line 97)
```python
@dataclass
class CollectionInfo:
    name: str; dimension: int; count: int; created_at: float; description: str = ""
```

**What it is**: Summary information about a collection, returned by `list_collections()`.

#### `DatabaseStats` (Line 101)
```python
@dataclass
class DatabaseStats:
    total_records: int; total_collections: int; dimension: int; 
    memory_usage_bytes: int; storage_path: str; embedding_model: str; db_file: str
```

**What it is**: Overall database statistics, returned by `stats()`.

### 4. ID Generator (Line 104)
```python
def generate_id(prefix="vec"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
```

**What it does**: Creates unique record IDs like `"vec_a1b2c3d4"` using UUID v4. The `hex[:8]` gives 8 hex characters = 4 billion possible values (collision probability is negligible).

### 5. Self-Test (Lines 120–152)
When run directly (`python ARCHITECTURE.py`), validates the schema by:
1. Creating a temp SQLite database
2. Running the schema creation
3. Inserting a test record with metadata
4. Verifying SELECT queries
5. Testing cascade delete
6. Creating and verifying dataclass instances

---

## Import Map

| Module | What It Imports | Why |
|--------|----------------|-----|
| `core/vector_store.py` | `VectorRecord`, `SearchResult`, `CollectionInfo`, `DatabaseStats`, `generate_id` | Build and return data model instances |
| `storage/database.py` | `SCHEMA_SQL`, `SQL_QUERIES` | Execute the schema and queries |
| `tests/*` | Various dataclasses | Create test fixtures |
