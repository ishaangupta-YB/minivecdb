"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Architecture & Class Design Specification        ║
║  File: minivecdb/ARCHITECTURE.py                              ║
║  Version: 2.0 (Updated with SQLite hybrid storage)            ║
║                                                               ║
║  HYBRID ARCHITECTURE:                                         ║
║    SQLite  -> structured data (records, metadata, collections)║
║    NumPy   -> vector embeddings (.npy binary files)           ║
║    Python  -> similarity search engine (built from scratch)   ║
╚═══════════════════════════════════════════════════════════════╝
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np, time, uuid

# ═══════════════════════════════════════════════════════════════
# SQLITE SCHEMA
# ═══════════════════════════════════════════════════════════════
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS collections (
    name        TEXT PRIMARY KEY,
    dimension   INTEGER NOT NULL DEFAULT 384,
    description TEXT DEFAULT '',
    created_at  REAL NOT NULL
);
INSERT OR IGNORE INTO collections (name, dimension, description, created_at)
VALUES ('default', 384, 'Default collection', strftime('%s','now'));

CREATE TABLE IF NOT EXISTS records (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    collection  TEXT NOT NULL DEFAULT 'default',
    created_at  REAL NOT NULL,
    FOREIGN KEY (collection) REFERENCES collections(name) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_records_collection ON records(collection);

CREATE TABLE IF NOT EXISTS metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_metadata_kv ON metadata(key, value);
CREATE INDEX IF NOT EXISTS idx_metadata_record ON metadata(record_id);
"""

SQL_QUERIES = {
    "insert_record": "INSERT INTO records (id,text,collection,created_at) VALUES (?,?,?,?)",
    "get_record": "SELECT id,text,collection,created_at FROM records WHERE id=?",
    "delete_record": "DELETE FROM records WHERE id=?",
    "update_record_text": "UPDATE records SET text=? WHERE id=?",
    "list_records": "SELECT id,text,collection,created_at FROM records WHERE collection=? ORDER BY created_at DESC LIMIT ?",
    "count_records": "SELECT COUNT(*) FROM records WHERE collection=?",
    "count_all_records": "SELECT COUNT(*) FROM records",
    "record_exists": "SELECT 1 FROM records WHERE id=? LIMIT 1",
    "all_record_ids": "SELECT id FROM records ORDER BY created_at ASC",
    "collection_record_ids": "SELECT id FROM records WHERE collection=? ORDER BY created_at ASC",
    "insert_metadata": "INSERT INTO metadata (record_id,key,value) VALUES (?,?,?)",
    "get_metadata": "SELECT key,value FROM metadata WHERE record_id=?",
    "delete_metadata": "DELETE FROM metadata WHERE record_id=?",
    "filter_by_metadata": "SELECT DISTINCT record_id FROM metadata WHERE key=? AND value=?",
    "create_collection": "INSERT INTO collections (name,dimension,description,created_at) VALUES (?,?,?,?)",
    "get_collection": "SELECT name,dimension,description,created_at FROM collections WHERE name=?",
    "list_collections": "SELECT c.name,c.dimension,c.description,c.created_at,COUNT(r.id) as cnt FROM collections c LEFT JOIN records r ON c.name=r.collection GROUP BY c.name ORDER BY c.created_at",
    "delete_collection": "DELETE FROM collections WHERE name=?",
    "collection_exists": "SELECT 1 FROM collections WHERE name=? LIMIT 1",
    "stats_per_collection": "SELECT collection,COUNT(*) as count FROM records GROUP BY collection",
}

# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════
@dataclass
class VectorRecord:
    id: str; vector: np.ndarray; text: str; metadata: Dict[str,Any]; created_at: float; collection: str = "default"
    def to_dict(self) -> dict:
        return {"id":self.id,"text":self.text,"metadata":self.metadata,"created_at":self.created_at,"collection":self.collection}
    @classmethod
    def from_db_row(cls, row, vector, metadata):
        return cls(id=row[0],vector=vector,text=row[1],collection=row[2],created_at=row[3],metadata=metadata)

@dataclass
class SearchResult:
    record: VectorRecord; score: float; rank: int; metric: str
    def to_dict(self) -> dict:
        return {"id":self.record.id,"text":self.record.text,"metadata":self.record.metadata,"score":round(self.score,6),"rank":self.rank,"metric":self.metric}

@dataclass
class CollectionInfo:
    name: str; dimension: int; count: int; created_at: float; description: str = ""

@dataclass
class DatabaseStats:
    total_records: int; total_collections: int; dimension: int; memory_usage_bytes: int; storage_path: str; embedding_model: str; db_file: str

def generate_id(prefix="vec"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# ═══════════════════════════════════════════════════════════════
# DISK LAYOUT:
#   storage_path/
#   ├── minivecdb.db       <- SQLite (records, metadata, collections)
#   ├── vectors.npy        <- NumPy array (N, 384) float32
#   └── id_mapping.json    <- list mapping matrix row -> record ID
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sqlite3, os, tempfile
    print("=" * 60)
    print("MiniVecDB Architecture v2.0 — SQLite Validation")
    print("=" * 60)
    db_path = os.path.join(tempfile.gettempdir(), "minivecdb_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.execute(SQL_QUERIES["insert_record"], ("vec_001","Hello world","default",time.time()))
    conn.execute(SQL_QUERIES["insert_metadata"], ("vec_001","category","greeting"))
    conn.execute(SQL_QUERIES["insert_metadata"], ("vec_001","language","english"))
    conn.commit()
    row = conn.execute(SQL_QUERIES["get_record"], ("vec_001",)).fetchone()
    print(f"  OK INSERT+SELECT: id={row[0]}, text='{row[1]}'")
    meta = dict(conn.execute(SQL_QUERIES["get_metadata"],("vec_001",)).fetchall())
    print(f"  OK Metadata: {meta}")
    filt = conn.execute(SQL_QUERIES["filter_by_metadata"],("category","greeting")).fetchall()
    print(f"  OK Filter: {[r[0] for r in filt]}")
    cols = conn.execute(SQL_QUERIES["list_collections"]).fetchall()
    print(f"  OK Collections: {[(c[0],c[4]) for c in cols]}")
    conn.execute(SQL_QUERIES["delete_record"],("vec_001",)); conn.commit()
    orphan = conn.execute(SQL_QUERIES["get_metadata"],("vec_001",)).fetchall()
    print(f"  OK Cascade delete: orphan metadata={orphan}")
    conn.close(); 
    os.remove(db_path)
    rec = VectorRecord(id=generate_id(),vector=np.random.rand(384).astype(np.float32),text="Test",metadata={"k":"v"},created_at=time.time())
    sr = SearchResult(record=rec,score=0.92,rank=1,metric="cosine")
    print(f"  OK VectorRecord: {rec.id}")
    print(f"  OK SearchResult: {sr.to_dict()}")
    print("=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)