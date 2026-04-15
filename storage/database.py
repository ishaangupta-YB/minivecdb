"""
+===============================================================+
|  MiniVecDB -- DatabaseManager (SQLite Wrapper, v3.0)           |
|  File: minivecdb/storage/database.py                           |
|                                                                |
|  Unified schema: one SQLite file holds sessions, conversations |
|  messages, collections, records, and metadata. Every record-  |
|  level operation is scoped to the session the manager was     |
|  bound to on construction.                                     |
|                                                                |
|  Public method signatures for record/metadata/collection work |
|  are kept compatible with pre-v3 callers (collection is still |
|  referenced by name); internally the session_id + composite   |
|  keys isolate each run.                                        |
+===============================================================+
"""

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple, Iterator

# ---------------------------------------------------------------
# Import schema + parameterised queries from ARCHITECTURE.py.
# ARCHITECTURE.py lives at the project root (one directory above
# storage/), so we prepend the parent directory to sys.path.
# ---------------------------------------------------------------
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ARCHITECTURE import SCHEMA_SQL, SQL_QUERIES, SessionInfo, MessageRow


class DatabaseManager:
    """
    SQLite access layer for MiniVecDB.

    One DatabaseManager instance is bound to exactly one session. All
    record / metadata / collection queries are scoped to that session
    via an internal `session_id`. Multiple sessions share the same
    physical .db file; the schema's foreign keys keep them isolated
    at the row level.

    Backward-compat note: when constructed without `session_name`, the
    manager derives a session from the parent directory of `db_path`
    so pre-v3 test fixtures (`DatabaseManager("/tmp/foo/minivecdb.db")`)
    keep working unchanged.
    """

    def __init__(
        self,
        db_path: str,
        session_name: Optional[str] = None,
        session_storage_path: Optional[str] = None,
    ) -> None:
        """Open the SQLite file and bind to (or create) a session row."""
        self.db_path: str = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

        # Derive session_name + storage_path defaults from db_path so
        # legacy callers keep working without changes.
        abs_db_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(abs_db_path)
        if session_name is None:
            derived = os.path.basename(db_dir) or "default"
            session_name = derived
        if session_storage_path is None:
            session_storage_path = db_dir

        self.session_name: str = session_name
        self.session_storage_path: str = os.path.abspath(session_storage_path)
        self.session_id: int = self._ensure_session(
            self.session_name, self.session_storage_path
        )
        self.conversation_id: int = self._get_or_create_default_conversation(
            self.session_id
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------
    @staticmethod
    def _require_non_empty_string(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    def _ensure_session(self, name: str, storage_path: str) -> int:
        """Upsert a session row and return its id."""
        now = time.time()
        self._conn.execute(
            SQL_QUERIES["upsert_session"],
            (name, storage_path, now, now),
        )
        self._conn.commit()
        row = self._conn.execute(
            SQL_QUERIES["get_session_by_name"], (name,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to register session '{name}'.")
        return int(row[0])

    def _get_or_create_default_conversation(self, session_id: int) -> int:
        """Return the session's first conversation id (trigger creates it)."""
        row = self._conn.execute(
            SQL_QUERIES["get_default_conversation_for_session"],
            (session_id,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                SQL_QUERIES["insert_conversation"],
                (session_id, "Default conversation", time.time()),
            )
            self._conn.commit()
            row = self._conn.execute(
                SQL_QUERIES["get_default_conversation_for_session"],
                (session_id,),
            ).fetchone()
        return int(row[0])

    def _resolve_collection_id(self, name: str) -> int:
        """Look up the collection id for the bound session; raise if missing."""
        row = self._conn.execute(
            SQL_QUERIES["get_collection_id_by_name"],
            (self.session_id, name),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Collection '{name}' does not exist in the current session."
            )
        return int(row[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run multiple writes atomically; rollback on any exception."""
        try:
            self._conn.execute("BEGIN")
            yield
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    # ===============================================================
    # RECORD CRUD (scoped to self.session_id)
    # ===============================================================

    def insert_record(
        self,
        id: str,
        text: str,
        collection: str,
        created_at: float,
        auto_commit: bool = True,
    ) -> None:
        """Insert a record into `collection` for the bound session."""
        self._require_non_empty_string(id, "id")
        self._require_non_empty_string(text, "text")
        self._require_non_empty_string(collection, "collection")

        collection_id = self._resolve_collection_id(collection)
        try:
            self._conn.execute(
                SQL_QUERIES["insert_record"],
                (id, self.session_id, collection_id, text, created_at),
            )
        except sqlite3.Error as exc:
            raise ValueError(f"Failed to insert record '{id}': {exc}") from exc

        if auto_commit:
            self._conn.commit()

    def get_record(self, id: str) -> Optional[Tuple]:
        """Return (id, text, collection_name, created_at) or None."""
        self._require_non_empty_string(id, "id")
        cursor = self._conn.execute(
            SQL_QUERIES["get_record"], (id, self.session_id)
        )
        return cursor.fetchone()

    def delete_record(self, id: str) -> bool:
        """Delete a record in the bound session; cascades to metadata."""
        self._require_non_empty_string(id, "id")
        cursor = self._conn.execute(
            SQL_QUERIES["delete_record"], (id, self.session_id)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_record_text(self, id: str, new_text: str) -> bool:
        """Update a record's text (vectors are the caller's problem)."""
        self._require_non_empty_string(id, "id")
        self._require_non_empty_string(new_text, "new_text")

        cursor = self._conn.execute(
            SQL_QUERIES["update_record_text"],
            (new_text, id, self.session_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def record_exists(self, id: str) -> bool:
        """Fast existence check scoped to the bound session."""
        self._require_non_empty_string(id, "id")
        cursor = self._conn.execute(
            SQL_QUERIES["record_exists"], (id, self.session_id)
        )
        return cursor.fetchone() is not None

    # ===============================================================
    # METADATA (EAV pattern; FK to records cascades)
    # ===============================================================

    def insert_metadata(
        self,
        record_id: str,
        key: str,
        value: str,
        auto_commit: bool = True,
    ) -> None:
        """Attach a key/value tag to a record."""
        self._require_non_empty_string(record_id, "record_id")
        self._require_non_empty_string(key, "key")

        try:
            self._conn.execute(
                SQL_QUERIES["insert_metadata"],
                (record_id, key, str(value)),
            )
        except sqlite3.Error as exc:
            raise ValueError(
                f"Failed to insert metadata '{key}' for record '{record_id}': {exc}"
            ) from exc

        if auto_commit:
            self._conn.commit()

    def get_metadata(self, record_id: str) -> Dict[str, str]:
        """Return {key: value} for a record (empty dict if none)."""
        self._require_non_empty_string(record_id, "record_id")
        cursor = self._conn.execute(
            SQL_QUERIES["get_metadata"], (record_id,)
        )
        return dict(cursor.fetchall())

    def delete_metadata(self, record_id: str) -> None:
        """Drop every metadata row tied to a record."""
        self._require_non_empty_string(record_id, "record_id")
        self._conn.execute(SQL_QUERIES["delete_metadata"], (record_id,))
        self._conn.commit()

    # Operators for advanced metadata filters. Numeric operators cast
    # the stored TEXT value to REAL so "2021" > 2020 compares as numbers.
    _FILTER_OPERATORS: Dict[str, str] = {
        "$gt":  "CAST(m.value AS REAL) > ?",
        "$lt":  "CAST(m.value AS REAL) < ?",
        "$gte": "CAST(m.value AS REAL) >= ?",
        "$lte": "CAST(m.value AS REAL) <= ?",
        "$ne":  "m.value != ?",
    }

    def filter_by_metadata(self, filters: Dict[str, object]) -> List[str]:
        """Return record IDs matching ALL filters (AND across keys)."""
        if not filters:
            return []

        result_sets: List[set] = []
        for key, value in filters.items():
            result_sets.append(self._execute_single_filter(key, value))

        matching = result_sets[0]
        for s in result_sets[1:]:
            matching &= s
        return sorted(matching)

    def _execute_single_filter(self, key: str, value: object) -> set:
        """Run one filter criterion; all queries scope via records.session_id."""
        if isinstance(value, str):
            cursor = self._conn.execute(
                SQL_QUERIES["filter_by_metadata"],
                (self.session_id, key, value),
            )
        elif isinstance(value, list):
            if len(value) == 0:
                return set()
            placeholders = ", ".join("?" for _ in value)
            sql = (
                "SELECT DISTINCT m.record_id "
                "FROM metadata m JOIN records r ON r.id = m.record_id "
                f"WHERE r.session_id = ? "
                "AND LOWER(TRIM(m.key)) = LOWER(TRIM(?)) "
                f"AND m.value IN ({placeholders})"
            )
            params = [self.session_id, key] + [str(v) for v in value]
            cursor = self._conn.execute(sql, params)
        elif isinstance(value, dict):
            conditions = ["r.session_id = ?", "LOWER(TRIM(m.key)) = LOWER(TRIM(?))"]
            params: list = [self.session_id, key]
            for op, operand in value.items():
                if op not in self._FILTER_OPERATORS:
                    raise ValueError(
                        f"Unknown filter operator '{op}'. "
                        f"Supported: {list(self._FILTER_OPERATORS.keys())}"
                    )
                conditions.append(self._FILTER_OPERATORS[op])
                if op == "$ne":
                    params.append(str(operand))
                else:
                    params.append(float(operand))

            sql = (
                "SELECT DISTINCT m.record_id "
                "FROM metadata m JOIN records r ON r.id = m.record_id "
                "WHERE " + " AND ".join(conditions)
            )
            cursor = self._conn.execute(sql, params)
        else:
            raise ValueError(
                f"Unsupported filter value type: {type(value).__name__}. "
                "Expected str, list, or dict."
            )

        return {row[0] for row in cursor.fetchall()}

    # ===============================================================
    # RECORD LISTING / ID RETRIEVAL
    # ===============================================================

    def get_record_ids_in_collection(self, collection: str) -> List[str]:
        """All record IDs in `collection`, oldest first."""
        self._require_non_empty_string(collection, "collection")
        cursor = self._conn.execute(
            SQL_QUERIES["collection_record_ids"],
            (self.session_id, collection),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_all_record_ids(self) -> List[str]:
        """Every record ID in the bound session."""
        cursor = self._conn.execute(
            SQL_QUERIES["all_record_ids"], (self.session_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_all_records_with_text(self) -> List[Tuple[str, str]]:
        """(id, text) for every record in the bound session."""
        cursor = self._conn.execute(
            SQL_QUERIES["all_records_with_text"], (self.session_id,)
        )
        return cursor.fetchall()

    def list_record_ids(
        self, collection: Optional[str] = None, limit: int = 100
    ) -> List[str]:
        """Record IDs, optionally filtered to one collection, up to `limit`."""
        if collection is not None:
            self._require_non_empty_string(collection, "collection")
            cursor = self._conn.execute(
                SQL_QUERIES["list_record_ids_in_collection"],
                (self.session_id, collection, limit),
            )
        else:
            cursor = self._conn.execute(
                SQL_QUERIES["list_record_ids"],
                (self.session_id, limit),
            )
        return [row[0] for row in cursor.fetchall()]

    def delete_records_in_collection(self, collection: str) -> int:
        """Delete every record in one collection of the bound session."""
        self._require_non_empty_string(collection, "collection")
        cursor = self._conn.execute(
            SQL_QUERIES["delete_records_in_collection"],
            (self.session_id, self.session_id, collection),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_all_records(self) -> int:
        """Wipe every record (and metadata, via cascade) in the bound session."""
        cursor = self._conn.execute(
            SQL_QUERIES["delete_all_records"], (self.session_id,)
        )
        self._conn.commit()
        return cursor.rowcount

    # ===============================================================
    # COLLECTION CRUD (session-scoped)
    # ===============================================================

    def create_collection(
        self,
        name: str,
        dimension: int = 384,
        description: str = "",
    ) -> None:
        """Create a new collection inside the bound session."""
        self._require_non_empty_string(name, "name")
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer.")

        try:
            self._conn.execute(
                SQL_QUERIES["create_collection"],
                (self.session_id, name, dimension, description, time.time()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise ValueError(
                f"Failed to create collection '{name}': {exc}"
            ) from exc

    def list_collections(self) -> List[Tuple]:
        """List (name, dimension, description, created_at, count) tuples.

        Preserves the pre-v3 column order so existing callers do not break.
        """
        cursor = self._conn.execute(
            SQL_QUERIES["list_collections_in_session"], (self.session_id,)
        )
        # Query returns (id, name, dimension, description, created_at, record_count).
        # Old shape was (name, dimension, description, created_at, count).
        return [
            (row[1], row[2], row[3], row[4], row[5])
            for row in cursor.fetchall()
        ]

    def delete_collection(self, name: str) -> bool:
        """Drop a collection (cascades to records + metadata)."""
        self._require_non_empty_string(name, "name")
        cursor = self._conn.execute(
            SQL_QUERIES["delete_collection"], (self.session_id, name)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def collection_exists(self, name: str) -> bool:
        """True when `name` exists inside the bound session."""
        self._require_non_empty_string(name, "name")
        cursor = self._conn.execute(
            SQL_QUERIES["collection_exists"], (self.session_id, name)
        )
        return cursor.fetchone() is not None

    # ===============================================================
    # STATISTICS
    # ===============================================================

    def count_records(self, collection: Optional[str] = None) -> int:
        """Number of records in the bound session (or one collection)."""
        if collection is None:
            cursor = self._conn.execute(
                SQL_QUERIES["count_all_records"], (self.session_id,)
            )
        else:
            self._require_non_empty_string(collection, "collection")
            cursor = self._conn.execute(
                SQL_QUERIES["count_records"],
                (self.session_id, collection),
            )
        return cursor.fetchone()[0]

    def stats_per_collection(self) -> Dict[str, int]:
        """Dict of {collection_name: record_count} for the bound session."""
        cursor = self._conn.execute(
            SQL_QUERIES["stats_per_collection"], (self.session_id,)
        )
        return dict(cursor.fetchall())

    # ===============================================================
    # SESSIONS / CONVERSATIONS / MESSAGES (new in v3.0)
    # ===============================================================

    def list_sessions(self) -> List[SessionInfo]:
        """Every session in the shared DB with derived message/record counts."""
        rows = self._conn.execute(SQL_QUERIES["list_sessions"]).fetchall()
        sessions: List[SessionInfo] = []
        for row in rows:
            session_id = int(row[0])
            (msg_count,) = self._conn.execute(
                SQL_QUERIES["count_messages_in_session"],
                (session_id,),
            ).fetchone()
            (record_count,) = self._conn.execute(
                SQL_QUERIES["count_records_in_session"],
                (session_id,),
            ).fetchone()

            sessions.append(
                SessionInfo(
                    id=session_id,
                    name=row[1],
                    storage_path=row[2],
                    created_at=row[3],
                    last_used_at=row[4],
                    msg_count=int(msg_count),
                    record_count=int(record_count),
                )
            )

        return sessions

    def log_message(
        self,
        kind: str,
        query_text: str,
        *,
        metric: Optional[str] = None,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
        result_count: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
        response_ref: Optional[str] = None,
        conversation_id: Optional[int] = None,
    ) -> int:
        """Persist one chat-history row; trigger bumps session.last_used_at."""
        if kind not in ("search", "insert"):
            raise ValueError("kind must be 'search' or 'insert'.")
        self._require_non_empty_string(query_text, "query_text")

        conv_id = conversation_id if conversation_id is not None else self.conversation_id
        cursor = self._conn.execute(
            SQL_QUERIES["insert_message"],
            (
                conv_id,
                kind,
                query_text,
                metric,
                top_k,
                category_filter,
                result_count,
                elapsed_ms,
                response_ref,
                time.time(),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def get_history(self, limit: int = 200) -> List[MessageRow]:
        """Chronological message list for the bound session."""
        rows = self._conn.execute(
            SQL_QUERIES["history_for_session"],
            (self.session_id, limit),
        ).fetchall()
        return [
            MessageRow(
                id=row[0],
                created_at=row[1],
                kind=row[2],
                query_text=row[3],
                metric=row[4],
                top_k=row[5],
                category_filter=row[6],
                result_count=row[7],
                elapsed_ms=row[8],
                response_ref=row[9],
            )
            for row in rows
        ]

    # ===============================================================
    # CONNECTION MANAGEMENT
    # ===============================================================

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
