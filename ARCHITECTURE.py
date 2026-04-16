"""
+===============================================================+
|  MiniVecDB -- Architecture & Class Design Specification       |
|  File: minivecdb/ARCHITECTURE.py                              |
|  Version: 3.0 (Unified DB: sessions + chat history + records) |
|                                                               |
|  HYBRID ARCHITECTURE:                                         |
|    SQLite  -> structured data, session-scoped                 |
|    NumPy   -> vector embeddings (.npy binary files)           |
|    Python  -> similarity search engine (built from scratch)   |
|                                                               |
|  One shared SQLite file holds every session's records,        |
|  metadata, collections, plus the app-level sessions,          |
|  conversations, and messages tables.                          |
|  Vectors stay per-session on disk (vectors.npy +              |
|  id_mapping.json in each session folder).                     |
+===============================================================+
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np, time, uuid

# ===============================================================
# SQLITE SCHEMA
# ===============================================================
# The schema is 3NF:
#   1NF — every column is atomic (no arrays, no JSON blobs).
#   2NF — every non-key attribute depends on the full PK (all
#         tables use a single-column surrogate PK).
#   3NF — no transitive dependencies (FKs are atomic references,
#         not derived values).
#
# Three triggers keep the app coherent:
#   - trg_create_default_conversation : every new session gets a
#     "Default conversation" row automatically.
#   - trg_create_default_collection   : every new session gets a
#     "default" collection so records can be inserted immediately.
#   - trg_touch_session_on_message    : every new message bumps
#     the session's last_used_at (uses a subquery to hop
#     conversation_id -> session_id).
# ===============================================================

SCHEMA_SQL = """
-- ----------------------------------------------------------------
-- 1) SESSIONS — one row per run folder under db_run/
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    storage_path  TEXT    NOT NULL,
    created_at    REAL    NOT NULL,
    last_used_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_used ON sessions(last_used_at DESC);

-- ----------------------------------------------------------------
-- 2) CONVERSATIONS — one default conversation per session
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    title       TEXT    NOT NULL DEFAULT 'Default conversation',
    created_at  REAL    NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

-- ----------------------------------------------------------------
-- 3) MESSAGES — user queries (search/insert) with response metadata
-- ----------------------------------------------------------------
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
CREATE INDEX IF NOT EXISTS idx_msg_conv    ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);

-- ----------------------------------------------------------------
-- 4) COLLECTIONS — session-scoped via composite UNIQUE
-- ----------------------------------------------------------------
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
CREATE INDEX IF NOT EXISTS idx_collections_session ON collections(session_id);

-- ----------------------------------------------------------------
-- 5) RECORDS — one row per stored document; FK to session+collection
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS records (
    id             TEXT    PRIMARY KEY,
    session_id     INTEGER NOT NULL,
    collection_id  INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    created_at     REAL    NOT NULL,
    FOREIGN KEY (session_id)    REFERENCES sessions(id)    ON DELETE CASCADE,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_records_session    ON records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_collection ON records(collection_id);

-- ----------------------------------------------------------------
-- 6) METADATA — EAV key/value tags on records
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metadata (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id  TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_metadata_kv     ON metadata(key, value);
CREATE INDEX IF NOT EXISTS idx_metadata_record ON metadata(record_id);

-- ----------------------------------------------------------------
-- TRIGGERS
-- ----------------------------------------------------------------
-- Every new session gets a Default conversation row.
CREATE TRIGGER IF NOT EXISTS trg_create_default_conversation
AFTER INSERT ON sessions
BEGIN
    INSERT INTO conversations (session_id, title, created_at)
    VALUES (NEW.id, 'Default conversation', NEW.created_at);
END;

-- Every new session gets a "default" collection row.
CREATE TRIGGER IF NOT EXISTS trg_create_default_collection
AFTER INSERT ON sessions
BEGIN
    INSERT INTO collections (session_id, name, dimension, description, created_at)
    VALUES (NEW.id, 'default', 384, 'Default collection', NEW.created_at);
END;

-- Every new message bumps its session's last_used_at. Uses a
-- subquery to resolve session_id from the message's conversation.
CREATE TRIGGER IF NOT EXISTS trg_touch_session_on_message
AFTER INSERT ON messages
BEGIN
    UPDATE sessions
       SET last_used_at = NEW.created_at
     WHERE id = (SELECT session_id FROM conversations WHERE id = NEW.conversation_id);
END;
"""

# ===============================================================
# SQL_QUERIES — central registry of every parameterised template.
# ===============================================================
# Every query uses ? placeholders. No string interpolation of user
# values anywhere. Record-level queries accept the session_id as a
# parameter, so the caller never leaks data across sessions.
# ===============================================================
SQL_QUERIES = {
    # ---- sessions -----------------------------------------------
    "insert_session": """
        INSERT INTO sessions (name, storage_path, created_at, last_used_at)
        VALUES (?, ?, ?, ?)
    """,
    "upsert_session": """
        INSERT INTO sessions (name, storage_path, created_at, last_used_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET storage_path = excluded.storage_path
    """,
    "get_session_by_name": """
        SELECT id, name, storage_path, created_at, last_used_at
          FROM sessions WHERE name = ?
    """,
    "touch_session": "UPDATE sessions SET last_used_at = ? WHERE id = ?",
    "list_sessions": """
        SELECT id, name, storage_path, created_at, last_used_at
          FROM sessions
         ORDER BY last_used_at DESC
    """,
    "count_messages_in_session": """
        SELECT COUNT(*)
          FROM messages m
          JOIN conversations c ON c.id = m.conversation_id
         WHERE c.session_id = ?
    """,
    "count_records_in_session": "SELECT COUNT(*) FROM records WHERE session_id = ?",

    # ---- conversations ------------------------------------------
    "get_default_conversation_for_session": """
        SELECT id FROM conversations
         WHERE session_id = ?
         ORDER BY created_at ASC
         LIMIT 1
    """,
    "insert_conversation": """
        INSERT INTO conversations (session_id, title, created_at) VALUES (?, ?, ?)
    """,

    # ---- messages -----------------------------------------------
    "insert_message": """
        INSERT INTO messages
            (conversation_id, kind, query_text, metric, top_k,
             category_filter, result_count, elapsed_ms, response_ref, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    # JOIN — /history page for a single session
    "history_for_session": """
        SELECT m.id, m.created_at, m.kind, m.query_text,
               m.metric, m.top_k, m.category_filter,
               m.result_count, m.elapsed_ms, m.response_ref
          FROM messages m
          JOIN conversations c ON c.id = m.conversation_id
         WHERE c.session_id = ?
         ORDER BY m.created_at ASC
         LIMIT ?
    """,

    # ---- collections (session-scoped) ---------------------------
    "create_collection": """
        INSERT INTO collections (session_id, name, dimension, description, created_at)
        VALUES (?, ?, ?, ?, ?)
    """,
    "get_collection_id_by_name": """
        SELECT id FROM collections WHERE session_id = ? AND name = ?
    """,
    "get_collection_full": """
        SELECT id, name, dimension, description, created_at
          FROM collections WHERE session_id = ? AND name = ?
    """,
    "list_collections_in_session": """
        SELECT c.id, c.name, c.dimension, c.description, c.created_at,
               COUNT(r.id) AS record_count
          FROM collections c
          LEFT JOIN records r ON r.collection_id = c.id
         WHERE c.session_id = ?
         GROUP BY c.id
         ORDER BY c.created_at
    """,
    "delete_collection": "DELETE FROM collections WHERE session_id = ? AND name = ?",
    "collection_exists": """
        SELECT 1 FROM collections WHERE session_id = ? AND name = ? LIMIT 1
    """,

    # ---- records (session-scoped) -------------------------------
    "insert_record": """
        INSERT INTO records (id, session_id, collection_id, text, created_at)
        VALUES (?, ?, ?, ?, ?)
    """,
    "get_record": """
        SELECT r.id, r.text, c.name, r.created_at
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.id = ? AND r.session_id = ?
    """,
    "delete_record": "DELETE FROM records WHERE id = ? AND session_id = ?",
    "update_record_text": "UPDATE records SET text = ? WHERE id = ? AND session_id = ?",
    "list_records": """
        SELECT r.id, r.text, c.name, r.created_at
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.session_id = ? AND c.name = ?
         ORDER BY r.created_at DESC
         LIMIT ?
    """,
    "count_records":     "SELECT COUNT(*) FROM records r JOIN collections c ON c.id = r.collection_id WHERE r.session_id = ? AND c.name = ?",
    "count_all_records": "SELECT COUNT(*) FROM records WHERE session_id = ?",
    "record_exists":     "SELECT 1 FROM records WHERE id = ? AND session_id = ? LIMIT 1",
    "all_record_ids":    "SELECT id FROM records WHERE session_id = ? ORDER BY created_at ASC",
    "collection_record_ids": """
        SELECT r.id
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.session_id = ? AND c.name = ?
         ORDER BY r.created_at ASC
    """,
    "all_records_with_text": """
        SELECT id, text FROM records WHERE session_id = ? ORDER BY created_at ASC
    """,
    "delete_records_in_collection": """
        DELETE FROM records WHERE session_id = ? AND collection_id = (
            SELECT id FROM collections WHERE session_id = ? AND name = ?
        )
    """,
    "delete_all_records": "DELETE FROM records WHERE session_id = ?",
    "list_record_ids": """
        SELECT id FROM records WHERE session_id = ? ORDER BY created_at ASC LIMIT ?
    """,
    "list_record_ids_in_collection": """
        SELECT r.id
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.session_id = ? AND c.name = ?
         ORDER BY r.created_at ASC
         LIMIT ?
    """,

    # ---- record browsing (session-scoped, with pagination) ------
    "browse_records_in_session": """
        SELECT r.id, r.text, c.name AS collection, r.created_at
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.session_id = ?
         ORDER BY r.created_at DESC
         LIMIT ? OFFSET ?
    """,
    "browse_records_in_collection": """
        SELECT r.id, r.text, c.name AS collection, r.created_at
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.session_id = ? AND c.name = ?
         ORDER BY r.created_at DESC
         LIMIT ? OFFSET ?
    """,
    "count_records_in_collection": """
        SELECT COUNT(*)
          FROM records r
          JOIN collections c ON c.id = r.collection_id
         WHERE r.session_id = ? AND c.name = ?
    """,

    # ---- metadata (scoped via JOIN on records.session_id) -------
    "insert_metadata": "INSERT INTO metadata (record_id, key, value) VALUES (?, ?, ?)",
    "get_metadata":    "SELECT key, value FROM metadata WHERE record_id = ?",
    "delete_metadata": "DELETE FROM metadata WHERE record_id = ?",
    # JOIN — scope the metadata filter to one session.
    "filter_by_metadata": """
        SELECT DISTINCT m.record_id
          FROM metadata m
          JOIN records  r ON r.id = m.record_id
                 WHERE r.session_id = ?
                     AND LOWER(TRIM(m.key)) = LOWER(TRIM(?))
                     AND LOWER(TRIM(m.value)) = LOWER(TRIM(?))
    """,

    # ---- stats / aggregates -------------------------------------
    "stats_per_collection": """
        SELECT c.name, COUNT(r.id) AS cnt
          FROM collections c
          LEFT JOIN records r ON r.collection_id = c.id
         WHERE c.session_id = ?
         GROUP BY c.name
    """,
}

# ===============================================================
# DATA MODELS
# ===============================================================
@dataclass
class VectorRecord:
    """One embedded document: id + text + metadata + vector."""
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]
    created_at: float
    collection: str = "default"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "collection": self.collection,
        }

    @classmethod
    def from_db_row(cls, row, vector, metadata):
        # row = (id, text, collection_name, created_at)
        return cls(
            id=row[0],
            vector=vector,
            text=row[1],
            collection=row[2],
            created_at=row[3],
            metadata=metadata,
        )


@dataclass
class SearchResult:
    """One ranked hit from VectorStore.search()."""
    record: VectorRecord
    score: float
    rank: int
    metric: str

    def to_dict(self) -> dict:
        return {
            "id": self.record.id,
            "text": self.record.text,
            "metadata": self.record.metadata,
            "score": round(self.score, 6),
            "rank": self.rank,
            "metric": self.metric,
        }


@dataclass
class CollectionInfo:
    """Per-session collection summary (name + record count)."""
    name: str
    dimension: int
    count: int
    created_at: float
    description: str = ""


@dataclass
class DatabaseStats:
    """Snapshot of the active session's store."""
    total_records: int
    total_collections: int
    dimension: int
    memory_usage_bytes: int
    storage_path: str
    embedding_model: str
    db_file: str
    session_name: str = ""


@dataclass
class SessionInfo:
    """One listed session with derived message/record counts."""
    id: int
    name: str
    storage_path: str
    created_at: float
    last_used_at: float
    msg_count: int
    record_count: int


@dataclass
class MessageRow:
    """One logged user query + its response metadata."""
    id: int
    created_at: float
    kind: str
    query_text: str
    metric: Optional[str]
    top_k: Optional[int]
    category_filter: Optional[str]
    result_count: Optional[int]
    elapsed_ms: Optional[float]
    response_ref: Optional[str]


def generate_id(prefix: str = "vec") -> str:
    """Generate a fresh, collision-resistant record ID like vec_a1b2c3d4."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ===============================================================
# DISK LAYOUT (v3.0):
#
#   project_root/
#   └── db_run/
#       ├── .active_run                    <- active run marker (folder name)
#       ├── minivecdb.db                   <- SHARED SQLite file (all sessions)
#       ├── model_cache/huggingface/       <- SentenceTransformer cache
#       └── <session_folder>/              <- e.g. demo_<ts>_<hex>
#           ├── vectors.npy                <- (N, 384) float32
#           └── id_mapping.json            <- row index -> record ID
#
# In legacy/test mode (VectorStore(storage_path=<path>)), the DB
# lives at <path>/minivecdb.db and the same folder holds the
# vector artefacts. Sessions table still gets populated — one
# session per test/tempdir — so the schema is identical.
# ===============================================================


if __name__ == "__main__":
    # Quick self-test: build an in-memory DB, run every trigger,
    # and exercise one JOIN + aggregate query.
    import sqlite3, os, tempfile

    print("=" * 60)
    print("MiniVecDB Architecture v3.0 — Schema Self-Test")
    print("=" * 60)

    db_path = os.path.join(tempfile.gettempdir(), "minivecdb_arch_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)

    # Insert a session -> triggers should create default conversation + collection.
    now = time.time()
    cur = conn.execute(SQL_QUERIES["insert_session"], ("demo_test", "/tmp/demo_test", now, now))
    session_id = cur.lastrowid
    conn.commit()

    (conv_count,) = conn.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,)).fetchone()
    (col_count,)  = conn.execute("SELECT COUNT(*) FROM collections  WHERE session_id = ?", (session_id,)).fetchone()
    assert conv_count == 1, "trigger_create_default_conversation failed"
    assert col_count  == 1, "trigger_create_default_collection failed"
    print(f"  OK Triggers 1+3: conv={conv_count}, col={col_count}")

    # Insert a record via the default collection.
    col_id = conn.execute(SQL_QUERIES["get_collection_id_by_name"], (session_id, "default")).fetchone()[0]
    rid = generate_id()
    conn.execute(SQL_QUERIES["insert_record"], (rid, session_id, col_id, "Hello world", now))
    conn.execute(SQL_QUERIES["insert_metadata"], (rid, "category", "greeting"))
    conn.commit()

    got = conn.execute(SQL_QUERIES["get_record"], (rid, session_id)).fetchone()
    assert got[0] == rid and got[1] == "Hello world" and got[2] == "default"
    print(f"  OK INSERT+JOIN: id={got[0]}, collection={got[2]}")

    # Insert a message -> trigger should bump last_used_at.
    conv_id = conn.execute(SQL_QUERIES["get_default_conversation_for_session"], (session_id,)).fetchone()[0]
    t_msg = now + 100
    conn.execute(
        SQL_QUERIES["insert_message"],
        (conv_id, "search", "hello", "cosine", 5, None, 3, 12.5, None, t_msg),
    )
    conn.commit()
    (touched,) = conn.execute("SELECT last_used_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert abs(touched - t_msg) < 1e-6, "trigger_touch_session_on_message failed"
    print(f"  OK Trigger 2: last_used_at advanced to {touched}")

    # Session list + per-session count queries.
    rows = conn.execute(SQL_QUERIES["list_sessions"]).fetchall()
    (msg_count,) = conn.execute(SQL_QUERIES["count_messages_in_session"], (session_id,)).fetchone()
    (record_count,) = conn.execute(SQL_QUERIES["count_records_in_session"], (session_id,)).fetchone()
    assert rows and msg_count == 1 and record_count == 1, "session count queries failed"
    print(f"  OK Session list+counts: {rows[0][1]} -> {msg_count} messages, {record_count} records")

    # Cascade delete.
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    (leftover,) = conn.execute("SELECT COUNT(*) FROM records").fetchone()
    (leftover_meta,) = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()
    (leftover_msg,)  = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
    assert leftover == 0 and leftover_meta == 0 and leftover_msg == 0, "cascade delete failed"
    print(f"  OK Cascade: all child rows gone after session delete")

    conn.close()
    os.remove(db_path)
    print("=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)
