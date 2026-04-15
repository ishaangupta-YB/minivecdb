# MiniVecDB — ER Diagram & Database Schema (v3.0)

## Entity-Relationship Diagram

The v3.0 SQLite schema uses **6 tables** connected by foreign keys with cascade deletes, plus **3 triggers** for automation:

```
┌──────────────────────────┐
│        sessions           │           ┌──────────────────────────────┐
│──────────────────────────│           │       conversations            │
│ PK id       INTEGER AUTO │──────1:N──│──────────────────────────────│
│    name     TEXT UNIQUE  │           │ PK id         INTEGER AUTO    │
│    storage_path TEXT     │           │ FK session_id  INTEGER        │
│    created_at   REAL     │           │    title       TEXT           │
│    last_used_at REAL     │           │    created_at  REAL           │
└──────────┬───────────────┘           └──────────────┬───────────────┘
           │                                          │
           │ 1:N                                      │ 1:N
           ▼                                          ▼
┌──────────────────────────┐           ┌──────────────────────────────┐
│      collections          │           │        messages                │
│──────────────────────────│           │──────────────────────────────│
│ PK id       INTEGER AUTO │           │ PK id              INT AUTO  │
│ FK session_id INTEGER    │           │ FK conversation_id INTEGER   │
│    name      TEXT        │           │    kind            TEXT      │
│    dimension  INTEGER    │           │    query_text      TEXT      │
│    description TEXT      │           │    metric          TEXT      │
│    created_at REAL       │           │    top_k           INTEGER   │
│  UNIQUE(session_id,name) │           │    category_filter TEXT      │
└──────────┬───────────────┘           │    result_count    INTEGER   │
           │                            │    elapsed_ms      REAL      │
           │ 1:N                        │    response_ref    TEXT      │
           ▼                            │    created_at      REAL      │
┌──────────────────────────┐           └──────────────────────────────┘
│       records             │
│──────────────────────────│
│ PK id       TEXT         │
│ FK session_id   INTEGER  │
│ FK collection_id INTEGER │
│    text         TEXT     │
│    created_at   REAL     │
└──────────┬───────────────┘
           │
           │ 1:N
           ▼
┌──────────────────────────┐
│       metadata            │
│──────────────────────────│
│ PK id       INTEGER AUTO │
│ FK record_id TEXT        │
│    key       TEXT        │
│    value     TEXT        │
└──────────────────────────┘
```

---

## Table Definitions (6 Tables)

### 1. `sessions` — One Row Per Run Folder

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    storage_path  TEXT    NOT NULL,
    created_at    REAL    NOT NULL,
    last_used_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_used ON sessions(last_used_at DESC);
```

**Purpose**: Tracks every session (run folder) in the system. The `name` is the folder name (e.g., `demo_1713052800_a1b2c3`) and is `UNIQUE`. The shared DB holds all sessions.

**Key points**:
- `last_used_at` is bumped automatically by `trg_touch_session_on_message` trigger
- Deleting a session cascades to ALL child rows (conversations, messages, collections, records, metadata)

### 2. `conversations` — One Default Per Session

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    title       TEXT    NOT NULL DEFAULT 'Default conversation',
    created_at  REAL    NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

**Purpose**: Every session gets one "Default conversation" auto-created by trigger. Conversations group messages. Designed for future multi-conversation support.

### 3. `messages` — Chat History (User Queries Only)

```sql
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL,
    kind             TEXT    NOT NULL CHECK (kind IN ('search','insert')),
    query_text       TEXT    NOT NULL,
    metric           TEXT    CHECK (metric IS NULL OR metric IN ('cosine','euclidean','dot')),
    top_k            INTEGER,
    category_filter  TEXT,
    result_count     INTEGER,
    elapsed_ms       REAL,
    response_ref     TEXT,
    created_at       REAL    NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

**Purpose**: Stores **user queries only** — never per-result rows. Response metadata (`result_count`, `elapsed_ms`, `response_ref`) sits on the same row. `kind` is CHECK-constrained to `'search'` or `'insert'`.

### 4. `collections` — Session-Scoped via Composite UNIQUE

```sql
CREATE TABLE IF NOT EXISTS collections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    dimension   INTEGER NOT NULL DEFAULT 384,
    description TEXT    DEFAULT '',
    created_at  REAL    NOT NULL,
    UNIQUE (session_id, name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

**Purpose**: Collections are now **session-scoped** — two different sessions can each have a collection named "papers" without collision. Uses a composite `UNIQUE(session_id, name)` constraint. Surrogate INTEGER PK (`id`) replaces the old TEXT PK.

### 5. `records` — One Row Per Stored Document

```sql
CREATE TABLE IF NOT EXISTS records (
    id             TEXT    PRIMARY KEY,
    session_id     INTEGER NOT NULL,
    collection_id  INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    created_at     REAL    NOT NULL,
    FOREIGN KEY (session_id)    REFERENCES sessions(id)    ON DELETE CASCADE,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);
```

**Purpose**: Stores the structured part of each record. The vector is in NumPy. Records now have **TWO foreign keys**: `session_id` for session-scoping and `collection_id` for collection membership (using the surrogate INTEGER PK from collections).

### 6. `metadata` — EAV Key/Value Tags

```sql
CREATE TABLE IF NOT EXISTS metadata (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id  TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);
```

**Purpose**: Same EAV pattern as before. `ON DELETE CASCADE` from records.

---

## Triggers (3)

### `trg_create_default_conversation`
```sql
AFTER INSERT ON sessions → INSERT INTO conversations (session_id, title, created_at)
```
Every new session automatically gets a "Default conversation" row.

### `trg_create_default_collection`
```sql
AFTER INSERT ON sessions → INSERT INTO collections (session_id, name, dimension, description, created_at)
```
Every new session automatically gets a `"default"` collection with dimension 384.

### `trg_touch_session_on_message`
```sql
AFTER INSERT ON messages →
  UPDATE sessions SET last_used_at = NEW.created_at
   WHERE id = (SELECT session_id FROM conversations WHERE id = NEW.conversation_id)
```
Uses a **subquery** to resolve `session_id` from the message's `conversation_id`, then bumps the session's `last_used_at`. This means the session picker always shows the most recently used sessions first.

---

## Indexes

| Index Name | Table | Columns | Purpose |
|-----------|-------|---------|---------|
| `idx_sessions_last_used` | sessions | (last_used_at DESC) | Fast ordering for session picker |
| `idx_conv_session` | conversations | (session_id) | Fast lookup: conversations for a session |
| `idx_msg_conv` | messages | (conversation_id) | Fast lookup: messages in a conversation |
| `idx_msg_created` | messages | (created_at) | Timeline ordering |
| `idx_collections_session` | collections | (session_id) | Fast lookup: collections in a session |
| `idx_records_session` | records | (session_id) | Session-scoped record queries |
| `idx_records_collection` | records | (collection_id) | Collection-filtered queries |
| `idx_metadata_kv` | metadata | (key, value) | Metadata filtering |
| `idx_metadata_record` | metadata | (record_id) | Get all metadata for a record |

---

## SQL Techniques Demonstrated

| Technique | Where Used | Query |
|-----------|-----------|-------|
| **LEFT JOIN** | `list_sessions_with_counts` | Sessions → Conversations → Messages for session picker |
| **INNER JOIN** | `get_record`, `list_records`, `history_for_session` | Records → Collections (name resolution), Messages → Conversations |
| **Subquery** | `trg_touch_session_on_message`, `list_sessions_with_counts` (record_count) | Resolves `session_id` via correlated/scalar subquery |
| **Aggregate / GROUP BY** | `list_sessions_with_counts`, `list_collections_in_session`, `stats_per_collection` | COUNT per session, per collection |
| **Triggers (3)** | Schema | Auto-create conversation + collection on session insert, auto-bump last_used_at |
| **CHECK constraint** | `messages.kind` | Enforces `kind IN ('search','insert')` |
| **Composite UNIQUE** | `collections` | `UNIQUE(session_id, name)` for session-scoped naming |
| **UPSERT (ON CONFLICT)** | `upsert_session` | `INSERT ... ON CONFLICT(name) DO UPDATE` for idempotent session registration |
| **CASCADE deletes** | All FK relationships | Deleting a session removes all its data across all 5 child tables |

---

## Cascade Delete Chain (v3.0)

```
DELETE session "demo_123"
  ├→ DELETE conversations WHERE session_id = X        (cascade from sessions)
  │    └→ DELETE messages WHERE conversation_id IN (...) (cascade from conversations)
  ├→ DELETE collections WHERE session_id = X           (cascade from sessions)
  │    └→ DELETE records WHERE collection_id IN (...)   (cascade from collections)
  │         └→ DELETE metadata WHERE record_id IN (...) (cascade from records)
  └→ (Also: records WHERE session_id = X — dual FK cascade)
```

---

## What Is NOT in SQLite

| Data | Storage | Why |
|------|---------|-----|
| Vector embeddings (384 floats per record) | `vectors.npy` (NumPy, per-session folder) | Fast batch matrix math |
| Row index → Record ID mapping | `id_mapping.json` (per-session folder) | Bridge between NumPy rows and SQLite records |
| Embedding model weights | `model_cache/huggingface/` | Large binary (~80MB), cached once |
