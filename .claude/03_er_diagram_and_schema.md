# MiniVecDB — ER Diagram & Database Schema

## Entity-Relationship Diagram

The SQLite database uses **3 tables** connected by foreign keys with cascade deletes:

```
┌─────────────────────┐           ┌─────────────────────────┐
│    collections      │           │       records            │
│─────────────────────│           │─────────────────────────│
│ PK name      TEXT   │◄──────────│ PK id         TEXT       │
│    dimension  INT   │  1:N FK   │    text       TEXT       │
│    description TEXT │  cascade  │ FK collection TEXT       │
│    created_at REAL  │           │    created_at REAL       │
└─────────────────────┘           └────────────┬────────────┘
                                               │
                                               │ 1:N FK
                                               │ cascade
                                               ▼
                                  ┌─────────────────────────┐
                                  │      metadata            │
                                  │─────────────────────────│
                                  │ PK id         INT AUTO   │
                                  │ FK record_id  TEXT       │
                                  │    key        TEXT       │
                                  │    value      TEXT       │
                                  └─────────────────────────┘
```

---

## Table Definitions

### 1. `collections` — Groups of Related Records

```sql
CREATE TABLE IF NOT EXISTS collections (
    name        TEXT PRIMARY KEY,     -- Unique collection identifier (e.g., "science_papers")
    dimension   INTEGER NOT NULL DEFAULT 384,  -- Vector dimensions (matches embedding model)
    description TEXT DEFAULT '',      -- Human-readable description
    created_at  REAL NOT NULL         -- Unix timestamp of creation
);
```

**Purpose**: Collections work like folders — they group related records. Every record belongs to exactly one collection. A `"default"` collection is auto-created via an `INSERT OR IGNORE` in the schema.

**Key points**:
- `name` is the PRIMARY KEY (no auto-increment integer — the name IS the unique identifier)
- `dimension` defaults to 384 (matching the all-MiniLM-L6-v2 model)
- The `"default"` collection is always present and cannot be deleted

### 2. `records` — The Core Data Table

```sql
CREATE TABLE IF NOT EXISTS records (
    id          TEXT PRIMARY KEY,     -- Unique record ID (e.g., "vec_a1b2c3d4")
    text        TEXT NOT NULL,        -- The original text that was embedded
    collection  TEXT NOT NULL DEFAULT 'default',  -- Which collection this belongs to
    created_at  REAL NOT NULL,        -- Unix timestamp
    FOREIGN KEY (collection) REFERENCES collections(name) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_records_collection ON records(collection);
```

**Purpose**: Stores the structured part of each record — the text, its collection membership, and creation time. The *vector* is NOT stored here (it's in NumPy).

**Key points**:
- `id` is a generated string like `"vec_a1b2c3d4"` (UUID v4 hex prefix)
- `ON DELETE CASCADE`: if a collection is deleted, all its records are automatically deleted
- The `idx_records_collection` index speeds up `WHERE collection=?` queries

### 3. `metadata` — EAV (Entity-Attribute-Value) Tags

```sql
CREATE TABLE IF NOT EXISTS metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-generated row ID
    record_id   TEXT NOT NULL,        -- Which record this tag belongs to
    key         TEXT NOT NULL,         -- Tag name (e.g., "category", "author")
    value       TEXT NOT NULL,         -- Tag value (e.g., "science", "Einstein")
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_metadata_kv ON metadata(key, value);
CREATE INDEX IF NOT EXISTS idx_metadata_record ON metadata(record_id);
```

**Purpose**: Stores arbitrary key-value tags for each record. Uses the **EAV (Entity-Attribute-Value) pattern** which allows different records to have completely different tags without altering the schema.

**Why EAV?**: One record might have `{"category": "science"}` while another has `{"language": "french", "topic": "cooking"}`. A fixed-column approach would need columns for every possible key. EAV is flexible.

**Key points**:
- `ON DELETE CASCADE`: if a record is deleted, all its metadata tags are automatically deleted
- `idx_metadata_kv` speeds up metadata filtering (`WHERE key=? AND value=?`)
- `idx_metadata_record` speeds up `WHERE record_id=?` lookups
- All values are stored as TEXT (numeric comparisons use `CAST(value AS REAL)`)

---

## Relationships

### collections → records (1:N)
- One collection can have many records
- Each record belongs to exactly one collection
- `ON DELETE CASCADE`: deleting a collection deletes all its records

### records → metadata (1:N)
- One record can have many metadata key-value pairs
- Each metadata row belongs to exactly one record
- `ON DELETE CASCADE`: deleting a record deletes all its metadata

### Cascade Delete Chain
Deleting a collection triggers a cascade:
```
DELETE collection "papers"
  └→ DELETE records WHERE collection = "papers"  (cascade from collections)
      └→ DELETE metadata WHERE record_id IN (...)  (cascade from records)
```

---

## Indexes

| Index Name | Table | Columns | Purpose |
|-----------|-------|---------|---------|
| `idx_records_collection` | records | (collection) | Fast lookup: "get all records in collection X" |
| `idx_metadata_kv` | metadata | (key, value) | Fast filter: "find records where category=science" |
| `idx_metadata_record` | metadata | (record_id) | Fast lookup: "get all metadata for record X" |

---

## SQL Queries (from ARCHITECTURE.py)

All queries are defined centrally in `SQL_QUERIES` dict using parameterised `?` placeholders:

### Record Operations
| Query Key | SQL | Purpose |
|-----------|-----|---------|
| `insert_record` | `INSERT INTO records (id,text,collection,created_at) VALUES (?,?,?,?)` | Add a new record |
| `get_record` | `SELECT id,text,collection,created_at FROM records WHERE id=?` | Fetch a record by ID |
| `delete_record` | `DELETE FROM records WHERE id=?` | Remove a record |
| `update_record_text` | `UPDATE records SET text=? WHERE id=?` | Change record text |
| `record_exists` | `SELECT 1 FROM records WHERE id=? LIMIT 1` | Check if record exists |
| `count_records` | `SELECT COUNT(*) FROM records WHERE collection=?` | Count records in collection |
| `count_all_records` | `SELECT COUNT(*) FROM records` | Count all records |

### Metadata Operations
| Query Key | SQL | Purpose |
|-----------|-----|---------|
| `insert_metadata` | `INSERT INTO metadata (record_id,key,value) VALUES (?,?,?)` | Add a tag |
| `get_metadata` | `SELECT key,value FROM metadata WHERE record_id=?` | Get all tags for a record |
| `delete_metadata` | `DELETE FROM metadata WHERE record_id=?` | Remove all tags |
| `filter_by_metadata` | `SELECT DISTINCT record_id FROM metadata WHERE key=? AND value=?` | Find matching records |

### Collection Operations
| Query Key | SQL | Purpose |
|-----------|-----|---------|
| `create_collection` | `INSERT INTO collections (name,dimension,description,created_at) VALUES (?,?,?,?)` | Create a collection |
| `list_collections` | `SELECT c.name,...,COUNT(r.id) FROM collections c LEFT JOIN records r...` | List with counts |
| `delete_collection` | `DELETE FROM collections WHERE name=?` | Delete a collection |
| `collection_exists` | `SELECT 1 FROM collections WHERE name=? LIMIT 1` | Check existence |

---

## PRAGMA Settings

```sql
PRAGMA foreign_keys = ON;  -- CRITICAL: Without this, CASCADE doesn't work!
```

SQLite *parses* `FOREIGN KEY` clauses by default but **ignores** them unless `PRAGMA foreign_keys = ON` is explicitly set. This is a common gotcha.

---

## What Is NOT in SQLite

The following data is stored outside of SQLite:

| Data | Storage | Why |
|------|---------|-----|
| Vector embeddings (384 floats per record) | `vectors.npy` (NumPy) | SQLite can't do fast matrix math; NumPy can |
| Row index → Record ID mapping | `id_mapping.json` | Links NumPy rows to SQLite records |
| Embedding model weights | `model_cache/huggingface/` | Large binary files (~80MB), cached once |
